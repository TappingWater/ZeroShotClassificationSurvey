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
from tokenizers import ByteLevelBPETokenizer


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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_classes = 100
val_classes = 50
test_classes = 50

proto_images_per_class = 5
train_captions_per_image = 10
proto_captions_per_class = "all"  # "all" or an integer as string like "16"

img_size = 224
max_len = 256
bpe_vocab_size = 8000
bpe_min_freq = 2

embed_dim = 256
text_layers = 4
text_heads = 4
text_ff_dim = 1024
dropout = 0.1

epochs = 30
batch_size = 512
lr = 5e-4
weight_decay = 1e-4
temperature = 0.07
num_workers = 1
use_amp = False


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


def train_bpe_tokenizer(texts: list, tok_dir: Path, vocab_size: int, min_freq: int):
    tok_dir.mkdir(parents=True, exist_ok=True)
    train_txt = tok_dir / "tokenizer_train.txt"
    with train_txt.open("w", encoding="utf-8") as f:
        for t in texts:
            t = (t or "").replace("\n", " ").strip()
            if t:
                f.write(t + "\n")

    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[str(train_txt)],
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )
    tok.save_model(str(tok_dir))

    tok = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))
    tok.add_special_tokens(["<pad>", "<unk>", "<bos>", "<eos>"])

    vocab = tok.get_vocab()
    pad_id = vocab.get("<pad>", 0)
    unk_id = vocab.get("<unk>", 1)
    bos_id = vocab.get("<bos>", 2)
    eos_id = vocab.get("<eos>", 3)
    return tok, pad_id, unk_id, bos_id, eos_id


def encode_batch(tok: ByteLevelBPETokenizer, texts: list, max_len: int, pad_id: int, bos_id: int, eos_id: int):
    ids_list = []
    mask_list = []
    for t in texts:
        t = (t or "").replace("\n", " ").strip()
        enc = tok.encode(t)
        ids = [bos_id] + enc.ids + [eos_id]
        ids = ids[:max_len]
        attn = [1] * len(ids)

        if len(ids) < max_len:
            pad_n = max_len - len(ids)
            ids = ids + [pad_id] * pad_n
            attn = attn + [0] * pad_n

        ids_list.append(ids)
        mask_list.append(attn)

    return torch.tensor(ids_list, dtype=torch.long), torch.tensor(mask_list, dtype=torch.long)


class SimpleCNN(nn.Module):
    def __init__(self, d: int):
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
        self.proj = nn.Linear(512, d)

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


class TextTransformer(nn.Module):
    def __init__(self, vocab_size: int, d: int, n_layers: int, n_heads: int, ff_dim: int, max_len: int, drop: float, pad_id: int):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=drop,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d)

    def forward(self, input_ids, attn_mask):
        b, l = input_ids.shape
        if l > self.max_len:
            input_ids = input_ids[:, : self.max_len]
            attn_mask = attn_mask[:, : self.max_len]
            l = self.max_len

        pos = torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        key_padding = (attn_mask == 0)
        x = self.encoder(x, src_key_padding_mask=key_padding)

        x = self.ln(x)
        attn = attn_mask.unsqueeze(-1).float()
        pooled = (x * attn).sum(dim=1) / attn.sum(dim=1).clamp_min(1.0)
        return F.normalize(pooled, dim=-1)


class ImageTextTrainDataset(Dataset):
    def __init__(self, hf_ds, records, image_col, image_tf, tokenizer, max_len, pad_id, bos_id, eos_id, k_caps):
        self.hf_ds = hf_ds
        self.records = records
        self.image_col = image_col
        self.image_tf = image_tf
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
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

        input_ids, attn_mask = encode_batch(self.tokenizer, [cap], self.max_len, self.pad_id, self.bos_id, self.eos_id)
        return img_t, input_ids[0], attn_mask[0], r["label_id"]


def clip_loss(img_emb, txt_emb, tau: float):
    logits = (img_emb @ txt_emb.t()) / tau
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)


def build_records(hf_ds, image_col, text_col, label_col):
    raw_labels = hf_ds[label_col]
    uniq = sorted(list(set(raw_labels)))
    label_map = {lab: i for i, lab in enumerate(uniq)}

    grouped = {}
    for i in range(len(hf_ds)):
        ex = hf_ds[i]
        raw_lab = ex[label_col]
        caps = normalize_captions(ex[text_col])

        key = i
        if key not in grouped:
            grouped[key] = {
                "hf_index": i,
                "label_raw": raw_lab,
                "label_id": label_map[raw_lab],
                "captions": [],
            }
        grouped[key]["captions"].extend(caps)

    records = []
    for rec in grouped.values():
        if rec["captions"]:
            rec["captions"] = list(dict.fromkeys([c.strip() for c in rec["captions"] if c.strip()]))
        records.append(rec)
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


@torch.no_grad()
def build_text_prototypes(txt_model, tok, hf_ds, records_proto, text_col, class_set, pad_id, bos_id, eos_id, max_len, proto_caps_per_class, device_):
    by_class_caps = defaultdict(list)
    for r in records_proto:
        cid = r["label_id"]
        if cid in class_set:
            by_class_caps[cid].extend(r["captions"])

    prototypes = {}
    for cid in sorted(list(class_set)):
        caps = by_class_caps.get(cid, [])
        if not caps:
            caps = [""]

        if proto_caps_per_class != "all":
            m = int(proto_caps_per_class)
            caps = caps[:m]

        bs = 64
        embs = []
        for i in range(0, len(caps), bs):
            chunk = caps[i : i + bs]
            input_ids, attn_mask = encode_batch(tok, chunk, max_len, pad_id, bos_id, eos_id)
            input_ids = input_ids.to(device_)
            attn_mask = attn_mask.to(device_)
            e = txt_model(input_ids, attn_mask)
            embs.append(e)
        embs = torch.cat(embs, dim=0)
        proto = F.normalize(embs.mean(dim=0), dim=-1)
        prototypes[cid] = proto.cpu()
    return prototypes


@torch.no_grad()
def zsl_eval(img_model, txt_protos, hf_ds, query_records, image_col, image_tf_eval, class_set, device_, bs=128):
    class_list = sorted(list(class_set))
    proto_mat = torch.stack([txt_protos[c] for c in class_list], dim=0).to(device_)

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

    records, label_map = build_records(hf_ds, image_col, text_col, label_col)
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
    }
    json_dump(split_meta, output_dir / "splits.json")

    train_texts = []
    for r in train_records:
        caps = r["captions"]
        if not caps:
            continue
        kk = len(caps) if train_captions_per_image <= 0 else min(train_captions_per_image, len(caps))
        train_texts.extend(caps[:kk])

    tok_dir = output_dir / "tokenizer"
    tokenizer, pad_id, unk_id, bos_id, eos_id = train_bpe_tokenizer(
        train_texts, tok_dir, vocab_size=bpe_vocab_size, min_freq=bpe_min_freq
    )
    vocab_size = len(tokenizer.get_vocab())

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

    train_ds = ImageTextTrainDataset(
        hf_ds=hf_ds,
        records=train_records,
        image_col=image_col,
        image_tf=image_tf_train,
        tokenizer=tokenizer,
        max_len=max_len,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        k_caps=train_captions_per_image,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    img_model = SimpleCNN(d=embed_dim).to(device)
    txt_model = TextTransformer(
        vocab_size=vocab_size,
        d=embed_dim,
        n_layers=text_layers,
        n_heads=text_heads,
        ff_dim=text_ff_dim,
        max_len=max_len,
        drop=dropout,
        pad_id=pad_id,
    ).to(device)

    opt = torch.optim.AdamW(list(img_model.parameters()) + list(txt_model.parameters()), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp))

    best_val_macro = -1.0
    best_path = output_dir / "best.pt"

    log_path = output_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    for epoch in range(1, epochs + 1):
        img_model.train()
        txt_model.train()

        running = 0.0
        n_steps = 0

        for imgs, input_ids, attn_mask, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(use_amp)):
                img_emb = img_model(imgs)
                txt_emb = txt_model(input_ids, attn_mask)
                loss = clip_loss(img_emb, txt_emb, temperature)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.item())
            n_steps += 1

        train_loss = running / max(1, n_steps)

        img_model.eval()
        txt_model.eval()

        val_protos = build_text_prototypes(
            txt_model, tokenizer, hf_ds, val_proto, text_col, val_c,
            pad_id, bos_id, eos_id, max_len, proto_captions_per_class, device
        )
        val_overall, val_macro = zsl_eval(
            img_model, val_protos, hf_ds, val_query, image_col, image_tf_eval, val_c, device, bs=min(256, batch_size)
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
                    "txt_model": txt_model.state_dict(),
                    "opt": opt.state_dict(),
                    "pad_id": pad_id,
                    "unk_id": unk_id,
                    "bos_id": bos_id,
                    "eos_id": eos_id,
                    "vocab_size": vocab_size,
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
    txt_model.load_state_dict(ckpt["txt_model"])
    img_model.to(device).eval()
    txt_model.to(device).eval()

    test_protos = build_text_prototypes(
        txt_model, tokenizer, hf_ds, test_proto, text_col, test_c,
        pad_id, bos_id, eos_id, max_len, proto_captions_per_class, device
    )
    test_overall, test_macro = zsl_eval(
        img_model, test_protos, hf_ds, test_query, image_col, image_tf_eval, test_c, device, bs=min(256, batch_size)
    )

    json_dump(
        {
            "best_epoch": int(ckpt["epoch"]),
            "best_val_macro": float(ckpt["best_val_macro"]),
            "test_overall_acc": float(test_overall),
            "test_macro_acc": float(test_macro),
        },
        output_dir / "final_metrics.json",
    )

    print(f"final unseen test | acc {test_overall:.4f} | macro {test_macro:.4f}")


if __name__ == "__main__":
    main()
