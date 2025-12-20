import argparse
import os
import json
import random
from pathlib import Path
from collections import defaultdict
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from torchvision import transforms
from torchvision.models import resnet101, ResNet101_Weights
from datasets import load_dataset
# Use raw transformers for better control
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions_per_image", type=int, default=-1, help="Number of captions to use per image. -1 for all.")
    parser.add_argument("--model_type", type=str, default="sbert", choices=["sbert", "random", "supervised"], help="Text encoder type.")
    parser.add_argument("--output_file", type=str, default="gan_results/gan_results.json", help="Path to save results JSON.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--train_classes", type=int, default=150, help="Number of training classes.")
    parser.add_argument("--val_classes", type=int, default=0, help="Number of validation classes.")
    parser.add_argument("--test_classes", type=int, default=50, help="Number of testing classes.")
    parser.add_argument("--num_seen_classes", type=int, default=-1, help="Override: Number of seen classes to use (subset of train_classes). -1 for all.")
    return parser.parse_args()

# CONFIG
# Use local cache relative to project or home ensures write access
scratch_root = Path("./cache").resolve()
hf_root = scratch_root / "hf_cache"

# put your paths here
os.environ["HF_HOME"] = str(hf_root)
os.environ["HF_DATASETS_CACHE"] = str(hf_root / "datasets")
os.environ["HF_HUB_CACHE"] = str(hf_root / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(hf_root / "transformers")
os.environ["TORCH_HOME"] = str(scratch_root / "torch_cache")

output_dir = Path("./gan_results").resolve()
output_dir.mkdir(parents=True, exist_ok=True)

dataset_name = "weiywang/CUB_200_2011_CAP"
seed = 61
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# Data Params
img_size = 224
# Classes are now dynamic via args

# GAN Params
z_dim = 100         # Noise dimension
embed_dim = 384     # MiniLM dimension
feature_dim = 2048  # ResNet101 penultimate layer
n_epochs = 50       # Reduced epochs as SBERT is strong
batch_size = 64
lr = 0.0001
beta1 = 0.5
cls_weight = 0.1    # Classification loss weight
lambda_gp = 10      # Gradient penalty lambda

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
         torch.cuda.manual_seed_all(s)

def get_random_embeddings(records, embed_dim):
    """Returns random embeddings for baseline."""
    print("Generating RANDOM text embeddings (Baseline)...")
    all_embs = []
    for r in records:
        n_caps = len(r["captions"])
        if n_caps == 0: n_caps = 1
        # Random noise
        embs = torch.randn(n_caps, embed_dim)
        all_embs.append(embs)
    return all_embs

# --- MODELS ---

class FeatureExtractor(nn.Module):
    """Pre-trained ResNet101 to extract 2048-dim features."""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(model.children())[:-1]) # Remove fc layer
        self.model.eval() # Freeze

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
        return x.view(x.size(0), -1)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.fc1 = nn.Linear(z_dim + embed_dim, 4096)
        self.fc2 = nn.Linear(4096, feature_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, noise, text_embedding):
        x = torch.cat([noise, text_embedding], dim=1)
        x = self.activation(self.fc1(x))
        x = self.relu(self.fc2(x)) # Features are typically > 0 after ReLU in ResNet? No, ResNet AvgPool output is >=0 because of previous ReLU.
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.fc1 = nn.Linear(feature_dim + embed_dim, 4096)
        self.fc2 = nn.Linear(4096, 1) # True / False
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, img_feature, text_embedding):
        x = torch.cat([img_feature, text_embedding], dim=1)
        x = self.activation(self.fc1(x))
        return self.fc2(x) # WGAN outputs scalar score (no sigmoid)

class Classifier(nn.Module):
    """Softmax Classifier."""
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.fc(x)
        return self.log_softmax(out)

# --- UTILS for WGAN-GP ---

def compute_gradient_penalty(D, real_samples, fake_samples, text_embeddings, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.rand((real_samples.size(0), 1)).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates, text_embeddings)
    fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty

# --- DATA ---

class FeatureDataset(Dataset):
    def __init__(self, features, labels, captions_emb):
        self.features = features
        self.labels = labels
        self.captions_emb = captions_emb
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return list of caption embeddings
        cap_emb = self.captions_emb[idx] # Tensor (N_caps, E)
        return self.features[idx], self.labels[idx], cap_emb

def extract_features(hf_ds, records, image_col, image_tf, device):
    """Runs data through ResNet101 to get features."""
    
    class RawDataset(Dataset):
        def __init__(self, hf_ds, records, image_col, image_tf):
            self.hf_ds = hf_ds
            self.records = records
            self.image_col = image_col
            self.image_tf = image_tf
        
        def __len__(self): return len(self.records)
        def __getitem__(self, idx):
            r = self.records[idx]
            ex = self.hf_ds[r["hf_index"]]
            img = ex[self.image_col]
            if img.mode != 'RGB': img = img.convert('RGB')
            return self.image_tf(img), r["label_id"]

    ds = RawDataset(hf_ds, records, image_col, image_tf)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    
    net = FeatureExtractor().to(device)
    
    feat_list = []
    label_list = []
    
    print("Extracting visual features...")
    # Iterate with progress
    for i, (imgs, labs) in enumerate(loader):
        if i % 10 == 0: print(f"Batch {i}/{len(loader)}")
        imgs = imgs.to(device)
        feats = net(imgs)
        feat_list.append(feats.cpu())
        label_list.extend(labs.tolist())
    
    return torch.cat(feat_list, dim=0), torch.tensor(label_list)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def precompute_text_embeddings(model_name, records, device, cache_folder):
    """
    Use AutoModel to encode all captions (SentenceTransformer logic manually).
    """
    print(f"Ensuring model download: {model_name}")
    try:
        # Explicitly download to get local path
        local_path = snapshot_download(repo_id=model_name, cache_dir=cache_folder)
        print(f"Model path: {local_path}")
    except Exception as e:
        print(f"Download warning: {e}. Trying to let AutoModel handle it.")
        local_path = model_name

    print(f"Loading AutoModel from: {local_path}")
    
    # Load with robust cache handling
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModel.from_pretrained(local_path)
    model.to(device)
    model.eval()
    
    all_embs = []
    
    print("Encoding text features...")
    
    all_texts = []
    boundaries = [] 
    
    for r in records:
        caps = r["captions"]
        caps = [c.strip() for c in caps if len(c.strip()) > 5]
        if not caps: caps = ["Uknown bird"]
        
        start = len(all_texts)
        all_texts.extend(caps)
        end = len(all_texts)
        boundaries.append((start, end))
    
    # Encode in chunks
    chunk_size = 256
    print(f"Total phrases to encode: {len(all_texts)}")
    
    with torch.no_grad():
        for i in range(0, len(all_texts), chunk_size):
            batch_texts = all_texts[i : i + chunk_size]
            
            # Tokenize
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
            
            # Compute token embeddings
            model_output = model(**encoded_input)
            
            # Perform pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            
            all_embs.append(sentence_embeddings.cpu())
            
            if i % 1024 == 0:
                print(f"Encoded {i}/{len(all_texts)}")

    # Flatten
    all_embs_tensor = torch.cat(all_embs, dim=0)
    
    # Re-group by record because we need list of tensors or tensor per record
    final_embs_list = []
    curr_idx = 0
    
    # Re-slice based on boundaries
    # Optimization: slice from the big tensor
    for start, end in boundaries:
        chunk = all_embs_tensor[start:end]
        final_embs_list.append(chunk)

    return final_embs_list

# --- MAIN ---

def main():
    args = parse_args()
    set_seed(seed)
    
    # 1. Load Data
    ds_dict = load_dataset(dataset_name)
    hf_ds = ds_dict['train']
    
    image_col = "image"
    text_col = "text"
    label_col = "label"

    # Reuse split logic
    from train_zsl_image import split_classes, build_records
    
    records, label_map = build_records(hf_ds, image_col, text_col, label_col)
    label_ids = [r["label_id"] for r in records]
    
    train_c, val_c, test_c = split_classes(label_ids, args.train_classes, args.val_classes, args.test_classes, seed)
    
    # Filter Train Classes if requested (Data Scaling Experiment)
    # If args.num_seen_classes is set and valid, use it to subset train_c
    if args.num_seen_classes > 0 and args.num_seen_classes < len(train_c):
        print(f"Subsetting Seen Classes: Using {args.num_seen_classes} out of {len(train_c)}...")
        sorted_train = sorted(list(train_c))
        # Take first N (Deterministic given seed)
        train_c = set(sorted_train[:args.num_seen_classes])
    
    train_indices = [i for i, r in enumerate(records) if r["label_id"] in train_c]
    test_indices = [i for i, r in enumerate(records) if r["label_id"] in test_c]

    # Limit captions per image IF requested (Simulation of "Side Info")
    if args.captions_per_image > 0:
        print(f"Limiting to {args.captions_per_image} captions per image...")
        for r in records:
            if len(r["captions"]) > args.captions_per_image:
                r["captions"] = r["captions"][:args.captions_per_image]
    
    # 2. Extract Visual Features
    image_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check if features cached
    feat_cache = output_dir / "resnet101_features.pt"
    if feat_cache.exists():
        print("Loading cached visual features...")
        cache = torch.load(feat_cache)
        all_features = cache["features"]
        all_labels_t = cache["labels"]
    else:
        all_features, all_labels_t = extract_features(hf_ds, records, image_col, image_tf, device)
        torch.save({"features": all_features, "labels": all_labels_t}, feat_cache)
    
    X_train = all_features[train_indices]
    y_train = all_labels_t[train_indices]
    
    # --- SUPERVISED UPPER BOUND ---
    if args.model_type == "supervised":
        print("--- Running Supervised Upper Bound on TEST Classes ---")
        # We want to see how well ResNet can classify the "Unseen" classes if it was allowed to see them.
        # So we take the data for 'test_classes' and split it 80/20.
        
        X_test_all = all_features[test_indices]
        y_test_all = all_labels_t[test_indices]
        
        # Create local 80/20 split
        n_samples = X_test_all.size(0)
        perm = torch.randperm(n_samples)
        n_train_sup = int(0.8 * n_samples)
        
        idx_train = perm[:n_train_sup]
        idx_test = perm[n_train_sup:]
        
        X_sup_train = X_test_all[idx_train]
        y_sup_train = y_test_all[idx_train]
        
        X_sup_test = X_test_all[idx_test]
        y_sup_test = y_test_all[idx_test]
        
        # Map labels to 0..49
        sorted_test_c = sorted(list(test_c))
        map_test = {c: i for i, c in enumerate(sorted_test_c)}
        
        y_sup_train = torch.tensor([map_test[l.item()] for l in y_sup_train])
        
        # Train Classifier
        print(f"Training Supervised Classifier on {len(X_sup_train)} real images of {len(test_c)} classes...")
        cls_net = Classifier(feature_dim, len(test_c)).to(device)
        cls_opt = torch.optim.Adam(cls_net.parameters(), lr=0.001)
        
        sup_ds = torch.utils.data.TensorDataset(X_sup_train.to(device), y_sup_train.to(device))
        sup_loader = DataLoader(sup_ds, batch_size=64, shuffle=True)
        
        for ep in range(30):
            for bx, by in sup_loader:
                cls_net.zero_grad()
                loss = F.nll_loss(cls_net(bx), by)
                loss.backward()
                cls_opt.step()
                
        # Evaluate
        cls_net.eval()
        correct = 0
        total = 0
        
        sup_test_ds = torch.utils.data.TensorDataset(X_sup_test.to(device), y_sup_test.to(device)) # y is Global ID
        # Wait, y_sup_test is global. We need to match prediction (0-49) to global (label_id) OR map y_sup_test to local
        
        test_loader = DataLoader(sup_test_ds, batch_size=32)
        with torch.no_grad():
            for bx, by in test_loader:
                out = cls_net(bx)
                preds = out.argmax(dim=1).cpu() # 0..49
                
                # Convert preds to global
                preds_global = [sorted_test_c[p] for p in preds]
                preds_global = torch.tensor(preds_global)
                
                correct += (preds_global == by.cpu()).sum().item()
                total += by.size(0)
                
        acc = correct / total
        print(f"Supervised Upper Bound Accuracy: {acc:.4f}")
        
        # Save and Exit
        res = {
            "zsl_gan_acc": acc,
            "model": "supervised_upper_bound",
            "feature_extractor": "resnet101"
        }
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out_path, "w"))
        return # Skip GAN part
    
    # 3. Text Embeddings
    if args.model_type == "random":
        all_cap_embs = get_random_embeddings(records, embed_dim)
    else:
        # Use MiniLM full path
        sbert_model = "sentence-transformers/all-MiniLM-L6-v2"
        all_cap_embs = precompute_text_embeddings(sbert_model, records, device, str(hf_root))

    
    train_cap_embs = [all_cap_embs[i] for i in train_indices]
    
    # 4. Pre-train a Classifier on REAL Image Features (Auxiliary)
    # This helps compute "Classification Loss" for the Generator
    # We map global train labels (e.g. 0, 5, 20...) to local (0..99)
    sorted_train_c = sorted(list(train_c))
    train_map_local = {c: i for i, c in enumerate(sorted_train_c)}
    
    y_train_local = torch.tensor([train_map_local[l.item()] for l in y_train])
    
    print("Pre-training auxiliary classifier on real features...")
    cls_aux = Classifier(feature_dim, len(train_c)).to(device)
    aux_opt = torch.optim.Adam(cls_aux.parameters(), lr=0.001)
    
    aux_ds = torch.utils.data.TensorDataset(X_train.to(device), y_train_local.to(device))
    aux_loader = DataLoader(aux_ds, batch_size=128, shuffle=True)
    
    for ep in range(10): # Quick train
        for bx, by in aux_loader:
            aux_opt.zero_grad()
            loss = F.nll_loss(cls_aux(bx), by)
            loss.backward()
            aux_opt.step()
    cls_aux.eval()
    
    # 5. GAN Setup
    def collate_fn(batch):
        features = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        captions = [item[2] for item in batch]
        return features, labels, captions

    dataset = FeatureDataset(X_train, y_train, train_cap_embs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    
    print("Starting GAN Training...")
    
    print("Starting GAN Training...")
    
    for epoch in range(args.epochs):
        d_losses = []
        g_losses = []
        
        for i, (imgs, labs, caps) in enumerate(dataloader):
            bs = imgs.size(0)
            
            # Pick one random caption embedding per image
            batch_txt = []
            for c_list in caps:
                if len(c_list) > 0:
                    idx = random.randint(0, len(c_list)-1)
                    batch_txt.append(c_list[idx])
                else:
                    batch_txt.append(torch.zeros(embed_dim))
            batch_txt = torch.stack(batch_txt).to(device)
            
            imgs = imgs.to(device)
            labs_local = torch.tensor([train_map_local[l.item()] for l in labs]).to(device)
            
            # --- Train Discriminator ---
            optD.zero_grad()
            
            real_validity = netD(imgs, batch_txt)
            d_loss_real = -torch.mean(real_validity)
            
            z = torch.randn(bs, z_dim).to(device)
            fake_imgs = netG(z, batch_txt)
            fake_validity = netD(fake_imgs, batch_txt)
            d_loss_fake = torch.mean(fake_validity)
            
            gp = compute_gradient_penalty(netD, imgs.data, fake_imgs.data, batch_txt.data, device)
            
            d_loss = d_loss_real + d_loss_fake + gp
            d_loss.backward()
            optD.step()
            d_losses.append(d_loss.item())
            
            # --- Train Generator ---
            if i % 5 == 0:
                optG.zero_grad()
                
                z = torch.randn(bs, z_dim).to(device)
                gen_imgs = netG(z, batch_txt)
                wgan_loss = -torch.mean(netD(gen_imgs, batch_txt))
                
                # Classification Loss (Enforce class consistency)
                pred_cls = cls_aux(gen_imgs) # LogSoftmax
                cls_loss = F.nll_loss(pred_cls, labs_local)
                
                loss_G = wgan_loss + (cls_weight * cls_loss)
                
                loss_G.backward()
                optG.step()
                g_losses.append(loss_G.item())
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: D_loss={np.mean(d_losses):.4f} G_loss={np.mean(g_losses):.4f}")

    # 6. Final Evaluation
    print("Training ZSL Classifier on Generated Features...")
    
    test_class_embs = {}
    for idx_in_rec, r in enumerate(records):
        lid = r["label_id"]
        if lid in test_c:
            if lid not in test_class_embs: test_class_embs[lid] = []
            test_class_embs[lid].append(all_cap_embs[idx_in_rec])
            
    syn_feats = []
    syn_labels = []
    items_per_class = 200 # Generate 200 samples per unseen class
    
    netG.eval()
    
    for c in test_c:
        if c not in test_class_embs: continue
        all_c = torch.cat(test_class_embs[c], dim=0) # (TotalCaps, E)
        proto = torch.mean(all_c, dim=0, keepdim=True).to(device) # Mean Prototype
        
        # Or better: Sample random captions from the class to capture diversity
        # But Proto is safer for Zero-Shot
        
        for _ in range(items_per_class // batch_size + 1):
             z = torch.randn(batch_size, z_dim).to(device)
             txt = proto.repeat(batch_size, 1) # Use prototype
             with torch.no_grad():
                 gen = netG(z, txt)
             syn_feats.append(gen.cpu())
             syn_labels.extend([c] * batch_size)
    
    X_syn = torch.cat(syn_feats, dim=0)
    y_syn = torch.tensor(syn_labels)
    
    sorted_test_c = sorted(list(test_c))
    map_test = {c: i for i, c in enumerate(sorted_test_c)}
    y_syn_mapped = torch.tensor([map_test[l.item()] for l in y_syn])
    
    cls_net = Classifier(feature_dim, len(test_c)).to(device)
    cls_opt = torch.optim.Adam(cls_net.parameters(), lr=0.001)
    
    syn_ds = torch.utils.data.TensorDataset(X_syn.to(device), y_syn_mapped.to(device))
    syn_loader = DataLoader(syn_ds, batch_size=64, shuffle=True)
    
    for ep in range(30):
        for bx, by in syn_loader:
             cls_net.zero_grad()
             loss = F.nll_loss(cls_net(bx), by)
             loss.backward()
             cls_opt.step()
             
    # Evaluate
    print("Evaluating on Unseen Classes...")
    X_test = all_features[test_indices]
    y_test = all_labels_t[test_indices]
    
    correct = 0
    total = 0
    cls_net.eval()
    
    test_ds = torch.utils.data.TensorDataset(X_test.to(device), y_test.to(device))
    test_loader = DataLoader(test_ds, batch_size=32)
    
    with torch.no_grad():
        for bx, by in test_loader:
             # by is global
             out = cls_net(bx)
             preds = out.argmax(dim=1).cpu()
             
             preds_global = [sorted_test_c[p] for p in preds]
             preds_global = torch.tensor(preds_global)
             
             correct += (preds_global == by.cpu()).sum().item()
             total += by.size(0)
             
    acc = correct / total
    print(f"ZSL Accuracy on Unseen Classes: {acc:.4f}")
    
    res = {
        "zsl_gan_acc": acc,
        "model": args.model_type,
        "captions_per_image": args.captions_per_image,
        "num_seen_classes": args.num_seen_classes,
        "feature_extractor": "resnet101"
    }
    
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_path, "w"))
    
    # Save Model Weights
    models_dir = out_path.parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f"{out_path.stem}.pt"
    
    print(f"Saving model weights to {model_path}...")
    torch.save({
        "netG": netG.state_dict(),
        "netD": netD.state_dict(),
        "cls_aux": cls_aux.state_dict(),
        "args": vars(args),
        "acc": acc
    }, model_path)

if __name__ == "__main__":
    main()
