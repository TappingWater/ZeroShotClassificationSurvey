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

output_dir = Path("./image_results").resolve()
output_dir.mkdir(parents=True, exist_ok=True)

dataset_name = "weiywang/CUB_200_2011_CAP"
dataset_split = None

seed = 61
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_classes = 100
val_classes = 50
test_classes = 50

proto_images_per_class = 5
train_captions_per_image = 10
proto_captions_per_class = "all"  # "all" or integer string

img_size = 224

embed_dim = 2048
epochs = 50
batch_size = 512
lr = 3e-3
weight_decay = 1e-4
margin = 0.2
num_workers = 8
use_amp = False

tfidf_min_df = 2
tfidf_max_features = 50000
use_bigrams = True


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def normalize_captions(val):
    if val is None:
        return []
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    if isinstance(val, (list, tuple)):
        out = []
        for x in val:
            if isinstance(x, str):
                xs = x.strip()
                if xs:
                    out.append(xs)
        return out
    if isinstance(val, dict):
        for k in ["text", "caption", "description", "sentence"]:
            if k in val and isinstance(val[k], str):
                s = val[k].strip()
                return [s] if s else []
    return []


def pil_to_rgb(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode != "RGB":
        return pil_img.convert("RGB")
    return pil_img


def detect_columns(example: dict):
    keys = set(example.keys())

    image_candidates = ["image", "img", "pixel_values"]
    text_candidates = ["text", "caption", "captions", "sentence", "sentences", "description", "descriptions"]
    label_candidates = ["label", "labels", "class", "class_id", "category", "target", "y"]

    def pick(cands):
        for c in cands:
            if c in keys:
                return c
        return None

    image_col = pick(image_candidates)
    text_col = pick(text_candidates)
    label_col = pick(label_candidates)

    if image_col is None or text_col is None or label_col is None:
        raise KeyError(
            f"could not detect required columns. found keys: {sorted(keys)}. need one each for image/text/label."
        )
    return image_col, text_col, label_col


def split_classes(label_ids, n_train, n_val, n_test, s):
    rng = random.Random(s)
    uniq = sorted(list(set(label_ids)))
    rng.shuffle(uniq)
    if n_train + n_val + n_test != len(uniq):
        raise ValueError(f"split sizes must sum to #classes. got {n_train+n_val+n_test} vs {len(uniq)}.")
    train_c = set(uniq[:n_train])
    val_c = set(uniq[n_train : n_train + n_val])
    test_c = set(uniq[n_train + n_val :])
    return train_c, val_c, test_c


def build_records(hf_ds, text_col, label_col):
    raw_labels = hf_ds[label_col]
    uniq = sorted(list(set(raw_labels)))
    label_map = {lab: i for i, lab in enumerate(uniq)}

    records = []
    for i in range(len(hf_ds)):
        ex = hf_ds[i]
        raw_lab = ex[label_col]
        caps = normalize_captions(ex[text_col])
        if caps:
            caps = list(dict.fromkeys([c.strip() for c in caps if c.strip()]))
        records.append(
            {
                "hf_index": i,
                "label_raw": raw_lab,
                "label_id": label_map[raw_lab],
                "captions": caps,
            }
        )
    return records, label_map


def split_proto_query(records, class_set, proto_per_class, s):
    rng = random.Random(s)
    by_class = defaultdict(list)
    for r in records:
        if r["label_id"] in class_set:
            by_class[r["label_id"]].append(r)

    proto = []
    query = []
    for cid, lst in by_class.items():
        rng.shuffle(lst)
        p = min(proto_per_class, len(lst))
        proto.extend(lst[:p])
        query.extend(lst[p:])
    return proto, query


def basic_tokenize(text: str):
    text = (text or "").lower()
    out = []
    cur = []
    for ch in text:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def build_vocab_and_idf(train_records, min_df: int, max_features: int, use_bigrams: bool):
    df = defaultdict(int)
    for r in train_records:
        caps = r["captions"]
        if not caps:
            continue
        kk = len(caps) if train_captions_per_image <= 0 else min(train_captions_per_image, len(caps))
        for cap in caps[:kk]:
            toks = basic_tokenize(cap)
            feats = set(toks)
            if use_bigrams and len(toks) >= 2:
                feats |= set([toks[i] + "_" + toks[i + 1] for i in range(len(toks) - 1)])
            for f in feats:
                df[f] += 1

    items = [(t, c) for t, c in df.items() if c >= min_df]
    items.sort(key=lambda x: x[1], reverse=True)
    if max_features is not None and len(items) > max_features:
        items = items[:max_features]

    vocab = {t: i for i, (t, _) in enumerate(items)}
    n_docs = 0
    for r in train_records:
        caps = r["captions"]
        if not caps:
            continue
        kk = len(caps) if train_captions_per_image <= 0 else min(train_captions_per_image, len(caps))
        n_docs += kk

    idf = np.zeros(len(vocab), dtype=np.float32)
    for t, i in vocab.items():
        idf[i] = np.log((1.0 + n_docs) / (1.0 + df[t])) + 1.0
    return vocab, idf


def tfidf_vector(text: str, vocab: dict, idf: np.ndarray, use_bigrams: bool):
    toks = basic_tokenize(text)
    idxs = []
    vals = []
    tf = defaultdict(int)
    for t in toks:
        if t in vocab:
            tf[vocab[t]] += 1
    if use_bigrams and len(toks) >= 2:
        for i in range(len(toks) - 1):
            bg = toks[i] + "_" + toks[i + 1]
            if bg in vocab:
                tf[vocab[bg]] += 1

    if not tf:
        return idxs, vals

    max_tf = max(tf.values())
    for i, c in tf.items():
        w = (c / max_tf) * float(idf[i])
        idxs.append(i)
        vals.append(w)

    v = np.array(vals, dtype=np.float32)
    n = float(np.linalg.norm(v)) if v.size else 0.0
    if n > 0:
        v /= n
    return idxs, v.tolist()


def make_text_proto_for_class(proto_records, cid, vocab, idf, use_bigrams: bool, proto_caps_per_class):
    caps = []
    for r in proto_records:
        if r["label_id"] == cid:
            caps.extend(r["captions"])
    if not caps:
        caps = [""]

    if proto_caps_per_class != "all":
        m = int(proto_caps_per_class)
        caps = caps[:m]

    accum = defaultdict(float)
    for cap in caps:
        idxs, vals = tfidf_vector(cap, vocab, idf, use_bigrams)
        for i, v in zip(idxs, vals):
            accum[i] += float(v)

    if not accum:
        return [], []

    idxs = sorted(accum.keys())
    vec = np.array([accum[i] for i in idxs], dtype=np.float32)
    n = float(np.linalg.norm(vec))
    if n > 0:
        vec /= n
    return idxs, vec.tolist()


class ImageBoWTrainDataset(Dataset):
    def __init__(self, hf_ds, records, image_col, text_col, image_tf, vocab, idf, use_bigrams, k_caps):
        self.hf_ds = hf_ds
        self.records = records
        self.image_col = image_col
        self.text_col = text_col
        self.image_tf = image_tf
        self.vocab = vocab
        self.idf = idf
        self.use_bigrams = use_bigrams
        self.k_caps = k_caps

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        ex = self.hf_ds[r["hf_index"]]
        img = pil_to_rgb(ex[self.image_col])
        img_t = self.image_tf(img)

        caps = r["captions"]
        if not caps:
            cap = ""
        else:
            kk = len(caps) if self.k_caps <= 0 else min(self.k_caps, len(caps))
            cap = random.choice(caps[:kk])

        idxs, vals = tfidf_vector(cap, self.vocab, self.idf, self.use_bigrams)
        return img_t, idxs, vals, r["label_id"]


def collate_bow(batch, vocab_size: int):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.tensor([b[3] for b in batch], dtype=torch.long)

    B = len(batch)
    X = torch.zeros(B, vocab_size, dtype=torch.float32)
    for i, (_, idxs, vals, _) in enumerate(batch):
        if idxs:
            X[i, torch.tensor(idxs, dtype=torch.long)] = torch.tensor(vals, dtype=torch.float32)
    return imgs, X, ys


class ImageEncoderToTextSpace(nn.Module):
    def __init__(self, out_dim: int):
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
        self.proj = nn.Linear(512, out_dim)

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
        z = self.proj(h)
        return F.normalize(z, dim=-1)


def cosine_margin_loss(img_z, txt_z, y, class_proto_mat, class_list, m: float):
    sims = img_z @ class_proto_mat.t()
    idx_map = {c: i for i, c in enumerate(class_list)}
    pos_idx = torch.tensor([idx_map[int(c)] for c in y.tolist()], device=img_z.device, dtype=torch.long)
    pos = sims.gather(1, pos_idx.unsqueeze(1)).squeeze(1)

    sims_masked = sims.clone()
    sims_masked[torch.arange(sims.size(0), device=sims.device), pos_idx] = -1e9
    neg = sims_masked.max(dim=1).values

    loss = F.relu(m - pos + neg).mean()
    return loss


@torch.no_grad()
def build_class_prototypes_sparse(proto_records, class_set, vocab, idf, use_bigrams, proto_caps_per_class):
    class_list = sorted(list(class_set))
    protos = {}
    for cid in class_list:
        idxs, vals = make_text_proto_for_class(proto_records, cid, vocab, idf, use_bigrams, proto_caps_per_class)
        protos[cid] = (idxs, vals)
    return protos


@torch.no_grad()
def dense_proto_matrix(protos_sparse, class_list, dim: int, device_):
    mat = torch.zeros(len(class_list), dim, dtype=torch.float32, device=device_)
    for j, cid in enumerate(class_list):
        idxs, vals = protos_sparse[cid]
        if idxs:
            mat[j, torch.tensor(idxs, dtype=torch.long, device=device_)] = torch.tensor(vals, dtype=torch.float32, device=device_)
    mat = F.normalize(mat, dim=-1)
    return mat


@torch.no_grad()
def zsl_eval_bow(img_model, proto_mat, class_list, hf_ds, query_records, image_col, image_tf_eval, class_set, device_, bs=128):
    idx_map = {c: i for i, c in enumerate(class_list)}

    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)
    overall_correct = 0
    overall_total = 0

    for i in range(0, len(query_records), bs):
        batch = query_records[i : i + bs]
        imgs = []
        labels = []
        for r in batch:
            ex = hf_ds[r["hf_index"]]
            img = pil_to_rgb(ex[image_col])
            imgs.append(image_tf_eval(img))
            labels.append(r["label_id"])

        imgs = torch.stack(imgs, dim=0).to(device_)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device_)

        img_emb = img_model(imgs)
        sims = img_emb @ proto_mat.t()
        pred_idx = sims.argmax(dim=1).tolist()
        preds = torch.tensor([class_list[j] for j in pred_idx], device=device_)

        correct = (preds == labels_t)
        overall_correct += int(correct.sum().item())
        overall_total += int(labels_t.numel())

        for y, ok in zip(labels, correct.tolist()):
            total_per_class[y] += 1
            if ok:
                correct_per_class[y] += 1

    overall_acc = overall_correct / max(1, overall_total)
    per_class_accs = []
    for c in class_list:
        denom = total_per_class.get(c, 0)
        if denom > 0:
            per_class_accs.append(correct_per_class.get(c, 0) / denom)
    macro_acc = float(np.mean(per_class_accs)) if per_class_accs else 0.0
    return overall_acc, macro_acc


def main():
    set_seed(seed)

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
    image_col, text_col, label_col = detect_columns(ex0)

    records, label_map = build_records(hf_ds, text_col, label_col)
    label_ids = [r["label_id"] for r in records]
    n_classes = len(set(label_ids))
    if n_classes != (train_classes + val_classes + test_classes):
        raise ValueError(
            f"expected {train_classes+val_classes+test_classes} total classes but dataset has {n_classes} unique labels."
        )

    train_c, val_c, test_c = split_classes(label_ids, train_classes, val_classes, test_classes, seed)

    train_records = [r for r in records if r["label_id"] in train_c]
    val_records = [r for r in records if r["label_id"] in val_c]
    test_records = [r for r in records if r["label_id"] in test_c]

    val_proto, val_query = split_proto_query(val_records, val_c, proto_images_per_class, seed + 1)
    test_proto, test_query = split_proto_query(test_records, test_c, proto_images_per_class, seed + 2)

    split_meta = {
        "dataset": dataset_name,
        "image_col": image_col,
        "text_col": text_col,
        "label_col": label_col,
        "seed": seed,
        "train_classes": sorted(list(train_c)),
        "val_classes": sorted(list(val_c)),
        "test_classes": sorted(list(test_c)),
        "proto_images_per_class": proto_images_per_class,
        "num_records": len(records),
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "num_test_records": len(test_records),
        "num_val_proto": len(val_proto),
        "num_val_query": len(val_query),
        "num_test_proto": len(test_proto),
        "num_test_query": len(test_query),
        "tfidf_min_df": tfidf_min_df,
        "tfidf_max_features": tfidf_max_features,
        "use_bigrams": use_bigrams,
    }
    json_dump(split_meta, output_dir / "splits_bow.json")

    vocab, idf = build_vocab_and_idf(train_records, min_df=tfidf_min_df, max_features=tfidf_max_features, use_bigrams=use_bigrams)
    vocab_size = len(vocab)
    if vocab_size == 0:
        raise RuntimeError("vocab is empty. reduce tfidf_min_df or increase caption usage.")

    image_tf_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image_tf_eval = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_ds = ImageBoWTrainDataset(
        hf_ds=hf_ds,
        records=train_records,
        image_col=image_col,
        text_col=text_col,
        image_tf=image_tf_train,
        vocab=vocab,
        idf=idf,
        use_bigrams=use_bigrams,
        k_caps=train_captions_per_image,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda b: collate_bow(b, vocab_size=vocab_size),
    )

    img_model = ImageEncoderToTextSpace(out_dim=vocab_size).to(device)
    opt = torch.optim.AdamW(img_model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp))

    best_val_macro = -1.0
    best_path = output_dir / "best_bow.pt"

    log_path = output_dir / "train_log_bow.jsonl"
    if log_path.exists():
        log_path.unlink()

    for epoch in range(1, epochs + 1):
        img_model.train()

        val_protos_sparse = build_class_prototypes_sparse(val_proto, val_c, vocab, idf, use_bigrams, proto_captions_per_class)
        val_class_list = sorted(list(val_c))
        val_proto_mat = dense_proto_matrix(val_protos_sparse, val_class_list, dim=vocab_size, device_=device)

        running = 0.0
        n_steps = 0

        for imgs, Xtxt, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            Xtxt = Xtxt.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(use_amp)):
                img_z = img_model(imgs)
                txt_z = F.normalize(Xtxt, dim=-1)
                loss = -torch.mean(torch.sum(img_z * txt_z, dim=1))

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.item())
            n_steps += 1

        train_loss = running / max(1, n_steps)

        img_model.eval()
        val_overall, val_macro = zsl_eval_bow(
            img_model, val_proto_mat, val_class_list, hf_ds, val_query, image_col, image_tf_eval, val_c, device, bs=min(256, batch_size)
        )

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_overall_acc": val_overall,
                "val_macro_acc": val_macro,
            }) + "\n")

        if val_macro > best_val_macro:
            best_val_macro = val_macro
            torch.save(
                {
                    "epoch": epoch,
                    "img_model": img_model.state_dict(),
                    "opt": opt.state_dict(),
                    "vocab": vocab,
                    "idf": idf,
                    "split_meta": split_meta,
                    "best_val_macro": best_val_macro,
                },
                best_path,
            )

        print(
            f"epoch {epoch:03d} | loss {train_loss:.4f} | "
            f"val zsl acc {val_overall:.4f} | val zsl macro {val_macro:.4f} | best {best_val_macro:.4f}"
        )

    ckpt = torch.load(best_path, map_location="cpu")
    img_model.load_state_dict(ckpt["img_model"])
    img_model.to(device).eval()

    vocab = ckpt["vocab"]
    idf = ckpt["idf"]
    vocab_size = len(vocab)

    test_protos_sparse = build_class_prototypes_sparse(test_proto, test_c, vocab, idf, use_bigrams, proto_captions_per_class)
    test_class_list = sorted(list(test_c))
    test_proto_mat = dense_proto_matrix(test_protos_sparse, test_class_list, dim=vocab_size, device_=device)

    test_overall, test_macro = zsl_eval_bow(
        img_model, test_proto_mat, test_class_list, hf_ds, test_query, image_col, image_tf_eval, test_c, device, bs=min(256, batch_size)
    )

    json_dump(
        {
            "best_epoch": int(ckpt["epoch"]),
            "best_val_macro": float(ckpt["best_val_macro"]),
            "test_overall_acc": float(test_overall),
            "test_macro_acc": float(test_macro),
            "vocab_size": int(vocab_size),
        },
        output_dir / "final_metrics_bow.json",
    )

    print(f"final unseen test | acc {test_overall:.4f} | macro {test_macro:.4f}")


if __name__ == "__main__":
    main()
