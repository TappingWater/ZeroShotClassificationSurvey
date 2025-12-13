import os
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from datasets import load_dataset


scratch_root = Path("/N/scratch/kisharma")
hf_root = scratch_root / "hf_cache"

os.environ["HF_HOME"] = str(hf_root)
os.environ["HF_DATASETS_CACHE"] = str(hf_root / "datasets")
os.environ["HF_HUB_CACHE"] = str(hf_root / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(hf_root / "transformers")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

output_dir = Path("./image_results").resolve()
output_dir.mkdir(parents=True, exist_ok=True)

dataset_name = "weiywang/CUB_200_2011_CAP"
dataset_split = None

seed = 61

train_classes = 100
val_classes = 50
test_classes = 50

val_to_train_frac = 0.70
test_to_train_frac = 0.70

img_size = 224
batch_size = 512
num_workers = 1

svm_max_iter = 8000
svm_c = 1.0
svm_dual = False


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def pil_to_rgb(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode != "RGB":
        return pil_img.convert("RGB")
    return pil_img


def detect_columns(example: dict):
    keys = set(example.keys())

    image_candidates = ["image", "img", "pixel_values"]
    label_candidates = ["label", "labels", "class", "class_id", "category", "target", "y"]

    def pick(cands):
        for c in cands:
            if c in keys:
                return c
        return None

    image_col = pick(image_candidates)
    label_col = pick(label_candidates)

    if image_col is None or label_col is None:
        raise KeyError(f"could not detect required columns. found keys: {sorted(keys)}. need image and label.")
    return image_col, label_col


def split_classes(class_ids, n_train, n_val, n_test, s):
    rng = random.Random(s)
    uniq = sorted(list(set(class_ids)))
    rng.shuffle(uniq)
    if n_train + n_val + n_test != len(uniq):
        raise ValueError(f"split sizes must sum to #classes. got {n_train+n_val+n_test} vs {len(uniq)}.")
    train_c = set(uniq[:n_train])
    val_c = set(uniq[n_train : n_train + n_val])
    test_c = set(uniq[n_train + n_val :])
    return train_c, val_c, test_c


class SimpleCNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            self._block(64, 128),
            self._block(128, 256),
            self._block(256, 512),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h = self.backbone(x).flatten(1)
        return h


class ImageIndexDataset(Dataset):
    def __init__(self, hf_ds, indices, image_col, label_col, label_to_id, image_tf):
        self.hf_ds = hf_ds
        self.indices = indices
        self.image_col = image_col
        self.label_col = label_col
        self.label_to_id = label_to_id
        self.image_tf = image_tf

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        ex = self.hf_ds[idx]
        img = pil_to_rgb(ex[self.image_col])
        x = self.image_tf(img)
        y = self.label_to_id[ex[self.label_col]]
        return x, y


@torch.no_grad()
def extract_features(backbone, loader, device_):
    backbone.eval()
    feats = []
    ys = []
    for x, y in loader:
        x = x.to(device_, non_blocking=True)
        f = backbone(x).detach().cpu().numpy()
        feats.append(f.astype(np.float32))
        ys.append(y.numpy().astype(np.int64))
    X = np.concatenate(feats, axis=0) if feats else np.zeros((0, 512), dtype=np.float32)
    y = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.int64)
    return X, y


def main():
    set_seed(seed)

    try:
        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise RuntimeError(f"sklearn is required for this script: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    if dataset_split is None:
        ds_dict = load_dataset(dataset_name)
        if isinstance(ds_dict, dict) and "train" in ds_dict:
            hf_ds = ds_dict["train"]
        elif isinstance(ds_dict, dict) and len(ds_dict.keys()) == 1:
            hf_ds = ds_dict[list(ds_dict.keys())[0]]
        elif isinstance(ds_dict, dict):
            hf_ds = ds_dict[list(ds_dict.keys())[0]]
        else:
            hf_ds = ds_dict
    else:
        hf_ds = load_dataset(dataset_name, split=dataset_split)

    ex0 = hf_ds[0]
    image_col, label_col = detect_columns(ex0)

    raw_labels = [hf_ds[i][label_col] for i in range(len(hf_ds))]
    uniq = sorted(list(set(raw_labels)))
    label_to_id = {lab: i for i, lab in enumerate(uniq)}
    label_ids = [label_to_id[lab] for lab in raw_labels]
    n_classes = len(uniq)

    if n_classes != (train_classes + val_classes + test_classes):
        raise ValueError(
            f"expected {train_classes+val_classes+test_classes} total classes but dataset has {n_classes} unique labels."
        )

    train_c, val_c, test_c = split_classes(label_ids, train_classes, val_classes, test_classes, seed)

    idx_by_class = defaultdict(list)
    for i, y in enumerate(label_ids):
        idx_by_class[int(y)].append(i)

    rng = random.Random(seed + 999)

    train_idx = []
    val_eval_idx = []
    test_eval_idx = []

    for cid in sorted(list(train_c)):
        train_idx.extend(idx_by_class[cid])

    for cid in sorted(list(val_c)):
        lst = idx_by_class[cid].copy()
        rng.shuffle(lst)
        k = int(len(lst) * val_to_train_frac)
        train_idx.extend(lst[:k])
        val_eval_idx.extend(lst[k:])

    for cid in sorted(list(test_c)):
        lst = idx_by_class[cid].copy()
        rng.shuffle(lst)
        k = int(len(lst) * test_to_train_frac)
        train_idx.extend(lst[:k])
        test_eval_idx.extend(lst[k:])

    split_meta = {
        "dataset": dataset_name,
        "image_col": image_col,
        "label_col": label_col,
        "seed": seed,
        "train_classes": sorted(list(train_c)),
        "val_classes": sorted(list(val_c)),
        "test_classes": sorted(list(test_c)),
        "val_to_train_frac": val_to_train_frac,
        "test_to_train_frac": test_to_train_frac,
        "num_train_images": len(train_idx),
        "num_val_eval_images": len(val_eval_idx),
        "num_test_eval_images": len(test_eval_idx),
        "svm_c": svm_c,
        "svm_max_iter": svm_max_iter,
        "svm_dual": svm_dual,
    }
    json_dump(split_meta, output_dir / "splits_supervised_svm.json")

    image_tf_eval = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_ds = ImageIndexDataset(hf_ds, train_idx, image_col, label_col, label_to_id, image_tf_eval)
    val_ds = ImageIndexDataset(hf_ds, val_eval_idx, image_col, label_col, label_to_id, image_tf_eval)
    test_ds = ImageIndexDataset(hf_ds, test_eval_idx, image_col, label_col, label_to_id, image_tf_eval)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    backbone = SimpleCNNBackbone().to(device)

    Xtr, ytr = extract_features(backbone, train_loader, device)
    Xva, yva = extract_features(backbone, val_loader, device)
    Xte, yte = extract_features(backbone, test_loader, device)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    svm = LinearSVC(C=svm_c, dual=svm_dual, max_iter=svm_max_iter)
    svm.fit(Xtr_s, ytr)

    val_acc = float((svm.predict(Xva_s) == yva).mean()) if len(yva) else 0.0
    test_acc = float((svm.predict(Xte_s) == yte).mean()) if len(yte) else 0.0

    with (output_dir / "svm_only_artifacts.npz").open("wb") as f:
        np.savez(
            f,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            svm_coef=svm.coef_,
            svm_intercept=svm.intercept_,
            label_names=np.array(uniq, dtype=object),
        )

    json_dump(
        {
            "val_acc": val_acc,
            "test_acc": test_acc,
            "feature_dim": int(Xtr.shape[1]) if Xtr.ndim == 2 else 0,
            "num_classes": int(n_classes),
        },
        output_dir / "final_metrics_supervised_svm_only.json",
    )

    print(f"final svm | val acc {val_acc:.4f} | test acc {test_acc:.4f}")


if __name__ == "__main__":
    main()
