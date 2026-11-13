"""Run several (variant, seed) jobs in parallel on one GPU.

    python colab/sweep.py --jobs 4 --seeds 0 1 2 [variant ...]

Without variants, runs every folder of colab/v. Jobs already in results.jsonl are skipped,
so the sweep can be restarted after a disconnect.
"""
import os, sys, json, argparse, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("variants", nargs="*")
a = ap.parse_args()
variants = a.variants or sorted(os.listdir(os.path.join(HERE, "v")))
res = os.path.join(HERE, "results.jsonl")
done = set()
if os.path.exists(res):
    for line in open(res):
        r = json.loads(line); done.add((r["variant"], r["seed"]))
todo = [(v, s) for s in a.seeds for v in variants if (v, s) not in done]

# Real-image Inception statistics once, before the parallel jobs.
sys.path[:0] = [HERE, ROOT + "/util"]; os.chdir(ROOT)
import torch, fastfid
class _M(torch.nn.Module):
    latent_dim = 1
    def __init__(self): super().__init__(); self.p = torch.nn.Parameter(torch.zeros(1))
    def encode(self, x): return torch.zeros(len(x), 1, device=x.device), None
    def decode(self, z): return torch.zeros(len(z), 1, 32, 32, device=z.device)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["FID_N"] = "50"; fastfid.evaluate(_M().to(dev)); del os.environ["FID_N"]
del fastfid; torch.cuda.empty_cache()

print(f"{len(todo)} jobs, {a.jobs} in parallel", flush=True)
running = []
t0 = time.time()
while todo or running:
    while todo and len(running) < a.jobs:
        v, s = todo.pop(0)
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "run.py"), v, str(s)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        running.append((v, s, p))
    time.sleep(2)
    for job in list(running):
        v, s, p = job
        if p.poll() is not None:
            out = p.stdout.read().strip().splitlines()
            print(f"[{(time.time()-t0)/60:5.1f} min] {out[-1] if out else ''}" if p.returncode == 0
                  else f"FAILED {v} {s}:\n" + "\n".join(out[-15:]), flush=True)
            running.remove(job)
print("ALL DONE", flush=True)
