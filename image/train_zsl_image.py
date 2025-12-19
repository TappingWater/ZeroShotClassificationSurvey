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
from torchvision.models import resnet18
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer


# CONFIG

scratch_root = Path("/N/scratch/kisharma")
hf_root = scratch_root / "hf_cache"

# put your paths here
os.environ["HF_HOME"] = str(hf_root)
os.environ["HF_DATASETS_CACHE"] = str(hf_root / "datasets")
os.environ["HF_HUB_CACHE"] = str(hf_root / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(hf_root / "transformers")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("TORCH_HOME", str(scratch_root / "torch_cache"))

output_dir = Path("./image_results").resolve()
output_dir.mkdir(parents=True, exist_ok=True)

dataset_name = "weiywang/CUB_200_2011_CAP"
dataset_split = None

seed = 61
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_classes = 100
val_classes = 50
test_classes = 50

# Training caption sampling (per image, after splitting into sentences)
train_captions_per_image = 10  # <=0 means "all available"

# Image / text preprocessing
img_size = 224
max_len = 256
bpe_vocab_size = 8000
bpe_min_freq = 2

# Model sizes
embed_dim = 256
dropout = 0.1

# Text encoder choice: "transformer", "rnn", "lstm"
TEXT_ENCODER_TYPE = "lstm"

# Transformer params (used only if TEXT_ENCODER_TYPE == "transformer")
text_layers = 4
text_heads = 4
text_ff_dim = 1024

# RNN/LSTM params (used only if TEXT_ENCODER_TYPE in {"rnn","lstm"})
rnn_hidden_dim = 256
rnn_layers = 2
rnn_bidirectional = True

# Optimization
epochs = 200
batch_size = 512
lr = 5e-4
weight_decay = 1e-4
temperature = 0.07
num_workers = 1
use_amp = False

# For debug 
EVAL_TEXT_SOURCE = "class_captions"  # or "classname_prompts"
EVAL_CAPTIONS_PER_CLASS = "all"      # "all" or an integer as string like "64"

# For ablation, not used for the presentation 
CLASSNAME_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the bird {}.",
    "a close-up photo of a {}.",
    "a photo of a {} in the wild.",
    "a bird called {}.",
    "{}.",
]



def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def json_dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def normalize_captions(val):
    
    def split_sentence_string(s):
        s = " ".join((s or "").replace("\n", " ").strip().split())
        if not s:
            return []
        parts = [p.strip() for p in s.split(".") if p.strip()]
        return [p + "." for p in parts]

    if val is None:
        return []

    if isinstance(val, str):
        return split_sentence_string(val)

    if isinstance(val, (list, tuple)):
        out = []
        for x in val:
            if isinstance(x, str):
                out.extend(split_sentence_string(x))
        return out

    if isinstance(val, dict):
        for k in ["text", "caption", "description", "sentence", "captions", "sentences", "descriptions"]:
            if k in val and isinstance(val[k], (str, list, tuple)):
                return normalize_captions(val[k])

    return []


def pil_to_rgb(pil_img):
    if pil_img.mode != "RGB":
        return pil_img.convert("RGB")
    return pil_img


def detect_columns(example):
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
    val_c = set(uniq[n_train: n_train + n_val])
    test_c = set(uniq[n_train + n_val:])
    return train_c, val_c, test_c


def train_bpe_tokenizer(texts, tok_dir, vocab_size, min_freq):
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


def encode_batch(tok, texts, max_len, pad_id, bos_id, eos_id):
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


class ResNet18Encoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        m = resnet18(weights=None)  # no pretrained weights, no download
        in_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.proj = nn.Linear(in_dim, d)

    def forward(self, x):
        h = self.backbone(x)
        z = self.proj(h)
        return F.normalize(z, dim=-1)


class TextTransformer(nn.Module):
    def __init__(self, vocab_size, d, n_layers, n_heads, ff_dim,
                 max_len, drop, pad_id):
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


class TextRNN(nn.Module):
    def __init__(self, vocab_size, d, hidden, n_layers,
                 bidir, max_len, drop, pad_id, cell):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d, padding_idx=pad_id)

        rnn_drop = drop if n_layers > 1 else 0.0
        if cell == "rnn":
            self.rnn = nn.RNN(
                input_size=d,
                hidden_size=hidden,
                num_layers=n_layers,
                batch_first=True,
                dropout=rnn_drop,
                bidirectional=bidir,
                nonlinearity="tanh",
            )
        elif cell == "lstm":
            self.rnn = nn.LSTM(
                input_size=d,
                hidden_size=hidden,
                num_layers=n_layers,
                batch_first=True,
                dropout=rnn_drop,
                bidirectional=bidir,
            )
        else:
            raise ValueError(f"unknown cell={cell}")

        out_dim = hidden * (2 if bidir else 1)
        self.proj = nn.Linear(out_dim, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, input_ids, attn_mask):
        b, l = input_ids.shape
        if l > self.max_len:
            input_ids = input_ids[:, : self.max_len]
            attn_mask = attn_mask[:, : self.max_len]
            l = self.max_len

        x = self.tok_emb(input_ids)  # (b,l,d)
        out, _ = self.rnn(x)         # (b,l,out_dim)

        attn = attn_mask.unsqueeze(-1).float()
        pooled = (out * attn).sum(dim=1) / attn.sum(dim=1).clamp_min(1.0)

        z = self.proj(pooled)
        z = self.ln(z)
        return F.normalize(z, dim=-1)


def build_text_encoder(vocab_size, pad_id):
    t = TEXT_ENCODER_TYPE.lower().strip()
    if t == "transformer":
        return TextTransformer(
            vocab_size=vocab_size,
            d=embed_dim,
            n_layers=text_layers,
            n_heads=text_heads,
            ff_dim=text_ff_dim,
            max_len=max_len,
            drop=dropout,
            pad_id=pad_id,
        )
    if t == "rnn":
        return TextRNN(
            vocab_size=vocab_size,
            d=embed_dim,
            hidden=rnn_hidden_dim,
            n_layers=rnn_layers,
            bidir=rnn_bidirectional,
            max_len=max_len,
            drop=dropout,
            pad_id=pad_id,
            cell="rnn",
        )
    if t == "lstm":
        return TextRNN(
            vocab_size=vocab_size,
            d=embed_dim,
            hidden=rnn_hidden_dim,
            n_layers=rnn_layers,
            bidir=rnn_bidirectional,
            max_len=max_len,
            drop=dropout,
            pad_id=pad_id,
            cell="lstm",
        )
    raise ValueError(f"Unknown TEXT_ENCODER_TYPE={TEXT_ENCODER_TYPE!r}")


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
            if self.k_caps <= 0:
                kk = len(caps)
            else:
                kk = min(self.k_caps, len(caps))
            cap = random.choice(caps[:kk])

        input_ids, attn_mask = encode_batch(self.tokenizer, [cap], self.max_len, self.pad_id, self.bos_id, self.eos_id)
        return img_t, input_ids[0], attn_mask[0], r["label_id"]


def build_records(hf_ds, image_col, text_col, label_col):
    raw_labels = hf_ds[label_col]
    uniq = sorted(list(set(raw_labels)))
    label_map = {lab: i for i, lab in enumerate(uniq)}

    grouped = {}
    for i in range(len(hf_ds)):
        ex = hf_ds[i]
        raw_lab = ex[label_col]
        caps = normalize_captions(ex[text_col])

        if i not in grouped:
            grouped[i] = {
                "hf_index": i,
                "label_raw": raw_lab,
                "label_id": label_map[raw_lab],
                "captions": [],
            }
        grouped[i]["captions"].extend(caps)

    records = []
    for rec in grouped.values():
        if rec["captions"]:
            rec["captions"] = list(dict.fromkeys([c.strip() for c in rec["captions"] if c.strip()]))
        records.append(rec)
    return records, label_map


def clip_loss(img_emb, txt_emb, tau):
    logits = (img_emb @ txt_emb.t()) / tau
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)


@torch.no_grad()
def build_classname_prototypes(txt_model, tok, id_to_label, class_set,
                               pad_id, bos_id, eos_id, max_len, device_):
    prototypes = {}
    for cid in sorted(list(class_set)):
        name = str(id_to_label[cid]).strip()
        texts = [tpl.format(name) for tpl in CLASSNAME_TEMPLATES]

        input_ids, attn_mask = encode_batch(tok, texts, max_len, pad_id, bos_id, eos_id)
        e = txt_model(input_ids.to(device_), attn_mask.to(device_))
        proto = F.normalize(e.mean(dim=0), dim=-1)
        prototypes[cid] = proto.cpu()
    return prototypes


@torch.no_grad()
def build_class_caption_prototypes(txt_model, tok, records, class_set,
                                   pad_id, bos_id, eos_id, max_len,
                                   captions_per_class, device_):
    by_class_caps = defaultdict(list)
    for r in records:
        cid = r["label_id"]
        if cid in class_set:
            by_class_caps[cid].extend(r["captions"])

    prototypes = {}
    for cid in sorted(list(class_set)):
        caps = by_class_caps.get(cid, [])
        if not caps:
            caps = [""]

        if captions_per_class != "all":
            m = int(captions_per_class)
            caps = caps[:m]

        bs = 64
        embs = []
        for i in range(0, len(caps), bs):
            chunk = caps[i: i + bs]
            input_ids, attn_mask = encode_batch(tok, chunk, max_len, pad_id, bos_id, eos_id)
            e = txt_model(input_ids.to(device_), attn_mask.to(device_))
            embs.append(e)

        embs = torch.cat(embs, dim=0)
        proto = F.normalize(embs.mean(dim=0), dim=-1)
        prototypes[cid] = proto.cpu()

    return prototypes


def make_class_prototypes(txt_model, tokenizer, id_to_label, records, class_set,
                          pad_id, bos_id, eos_id, max_len, device_):
    if EVAL_TEXT_SOURCE == "classname_prompts":
        return build_classname_prototypes(
            txt_model, tokenizer, id_to_label, class_set,
            pad_id, bos_id, eos_id, max_len, device_
        )
    if EVAL_TEXT_SOURCE == "class_captions":
        return build_class_caption_prototypes(
            txt_model, tokenizer, records, class_set,
            pad_id, bos_id, eos_id, max_len, EVAL_CAPTIONS_PER_CLASS, device_
        )
    raise ValueError(f"Unknown EVAL_TEXT_SOURCE={EVAL_TEXT_SOURCE!r}")


@torch.no_grad()
def zsl_eval(img_model, class_protos, hf_ds, eval_records,
             image_col, image_tf_eval, class_set, device_, bs=128):
    class_list = sorted(list(class_set))
    proto_mat = torch.stack([class_protos[c] for c in class_list], dim=0).to(device_)

    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)
    overall_correct = 0
    overall_total = 0

    for i in range(0, len(eval_records), bs):
        batch = eval_records[i: i + bs]
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

    # Load dataset
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

    # Detect columns
    ex0 = hf_ds[0]
    image_col, text_col, label_col = detect_columns(ex0)

    # Build per-image records (captions split into sentence chunks)
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

    id_to_label = {i: lab for lab, i in label_map.items()}

    # Save split metadata
    split_meta = {
        "dataset": dataset_name,
        "image_col": image_col,
        "text_col": text_col,
        "label_col": label_col,
        "seed": seed,
        "train_classes": sorted(list(train_c)),
        "val_classes": sorted(list(val_c)),
        "test_classes": sorted(list(test_c)),
        "num_records": len(records),
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "num_test_records": len(test_records),
        "eval_text_source": EVAL_TEXT_SOURCE,
        "eval_captions_per_class": EVAL_CAPTIONS_PER_CLASS if EVAL_TEXT_SOURCE == "class_captions" else None,
        "classname_templates": CLASSNAME_TEMPLATES if EVAL_TEXT_SOURCE == "classname_prompts" else None,
        "text_encoder_type": TEXT_ENCODER_TYPE,
    }
    json_dump(split_meta, output_dir / "splits.json")

    # Train tokenizer on training captions only
    train_texts = []
    for r in train_records:
        caps = r["captions"]
        if not caps:
            continue
        if train_captions_per_image <= 0:
            kk = len(caps)
        else:
            kk = min(train_captions_per_image, len(caps))
        train_texts.extend(caps[:kk])

    tok_dir = output_dir / "tokenizer"
    tokenizer, pad_id, unk_id, bos_id, eos_id = train_bpe_tokenizer(
        train_texts, tok_dir, vocab_size=bpe_vocab_size, min_freq=bpe_min_freq
    )
    vocab_size = len(tokenizer.get_vocab())

    # Transforms
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

    # Train loader
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

    # Models
    img_model = ResNet18Encoder(d=embed_dim).to(device)
    txt_model = build_text_encoder(vocab_size=vocab_size, pad_id=pad_id).to(device)

    opt = torch.optim.AdamW(
        list(img_model.parameters()) + list(txt_model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp))

    best_val_macro = -1.0
    best_path = output_dir / "best.pt"

    log_path = output_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    # Training loop
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

        # ZSL validation (unseen val classes)
        img_model.eval()
        txt_model.eval()

        val_protos = make_class_prototypes(
            txt_model, tokenizer, id_to_label, val_records, val_c,
            pad_id, bos_id, eos_id, max_len, device
        )
        val_overall, val_macro = zsl_eval(
            img_model, val_protos, hf_ds, val_records,
            image_col, image_tf_eval, val_c,
            device, bs=min(256, batch_size)
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
                    "text_encoder_type": TEXT_ENCODER_TYPE,
                },
                best_path,
            )

        print(
            f"epoch {epoch:03d} | loss {train_loss:.4f} | "
            f"val zsl acc {val_overall:.4f} | val zsl macro {val_macro:.4f} | best {best_val_macro:.4f}"
        )

    # Load best model
    ckpt = torch.load(best_path, map_location="cpu")
    img_model.load_state_dict(ckpt["img_model"])
    txt_model.load_state_dict(ckpt["txt_model"])
    img_model.to(device).eval()
    txt_model.to(device).eval()

    # ZSL test (unseen test classes)
    test_protos = make_class_prototypes(
        txt_model, tokenizer, id_to_label, test_records, test_c,
        pad_id, bos_id, eos_id, max_len, device
    )
    test_overall, test_macro = zsl_eval(
        img_model, test_protos, hf_ds, test_records,
        image_col, image_tf_eval, test_c,
        device, bs=min(256, batch_size)
    )

    json_dump(
        {
            "best_epoch": int(ckpt["epoch"]),
            "best_val_macro": float(ckpt["best_val_macro"]),
            "test_overall_acc": float(test_overall),
            "test_macro_acc": float(test_macro),
            "eval_text_source": EVAL_TEXT_SOURCE,
            "eval_captions_per_class": EVAL_CAPTIONS_PER_CLASS if EVAL_TEXT_SOURCE == "class_captions" else None,
            "text_encoder_type": TEXT_ENCODER_TYPE,
        },
        output_dir / "final_metrics.json",
    )

    print(f"final unseen test | acc {test_overall:.4f} | macro {test_macro:.4f}")


if __name__ == "__main__":
    main()
