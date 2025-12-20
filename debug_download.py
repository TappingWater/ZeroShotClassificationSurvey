import os
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel

# Setup env same as main script
scratch_root = Path("./cache").resolve()
hf_root = scratch_root / "hf_cache"

os.environ["HF_HOME"] = str(hf_root)
os.environ["HF_DATASETS_CACHE"] = str(hf_root / "datasets")
os.environ["HF_HUB_CACHE"] = str(hf_root / "hub")

model_id = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Attempting to download {model_id} using snapshot_download...")
try:
    path = snapshot_download(repo_id=model_id, cache_dir=hf_root)
    print(f"Success! Model downloaded to: {path}")
    
    print("Attempting to load AutoModel from local path...")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    print("Success! Model loaded.")
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
