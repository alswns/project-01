"""Train one variant with one seed and append its FID / reconstruction error to results.jsonl.

    python colab/run.py <variant> <seed>

Files in colab/v/<variant>/ shadow the ones in code/.
"""
import os, sys, time, json
import warnings; warnings.filterwarnings("ignore")
import torch
from torch.utils.data import DataLoader
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
variant = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
sys.path[:0] = [os.path.join(HERE, "v", variant), HERE, ROOT + "/code", ROOT + "/util"]
os.chdir(ROOT)
from mymodel import VAE
from myoptim import BATCH_SIZE, NUM_EPOCHS, get_optimizer
from mytrain import train_one_epoch
from data import get_dataset
from fastfid import evaluate
torch.manual_seed(seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = DataLoader(get_dataset("train"), batch_size=BATCH_SIZE, shuffle=True)
model = VAE().to(dev); opt = get_optimizer(model)
t = time.time()
for e in range(NUM_EPOCHS):
    l = train_one_epoch(model, opt, loader, dev)
tt = time.time() - t
r = evaluate(model)
out = dict(variant=variant, seed=seed, fid=round(r["fid"], 2), rec=round(r["reconstruction_error"], 2),
           loss=round(l["loss"], 1), recon=round(l["recon"], 1), kl=round(l["kl"], 1),
           params=sum(p.numel() for p in model.parameters()), train_s=round(tt), total_s=round(time.time() - t),
           device=torch.cuda.get_device_name() if dev.type == "cuda" else "cpu")
print(json.dumps(out), flush=True)
with open(os.path.join(HERE, "results.jsonl"), "a") as f:
    f.write(json.dumps(out) + "\n")
