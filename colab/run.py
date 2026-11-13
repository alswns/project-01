"""Train one variant with one seed and append its FID / reconstruction error to results.jsonl.

    python colab/run.py <variant> <seed>

Files in colab/v/<variant>/ shadow the ones in code/.
"""
import os, sys, time, json
import warnings; warnings.filterwarnings("ignore")
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
variant = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
sys.path[:0] = [os.path.join(HERE, "v", variant), HERE, ROOT + "/code", ROOT + "/util"]
os.chdir(ROOT)
from mymodel import VAE
import myoptim
from myoptim import BATCH_SIZE, NUM_EPOCHS, get_optimizer
from mytrain import train_one_epoch, loss_function
from data import get_dataset
from fastfid import evaluate
torch.manual_seed(seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True


class GPULoader:
    """Shuffled mini-batches of a dataset kept on the device (same as DataLoader(shuffle=True))."""
    def __init__(self, x, bs): self.x, self.bs = x, bs
    def __len__(self): return (len(self.x) + self.bs - 1) // self.bs
    def __iter__(self):
        idx = torch.randperm(len(self.x), device=self.x.device)
        for i in range(0, len(self.x), self.bs):
            yield self.x[idx[i:i + self.bs]]


loader = GPULoader(get_dataset("train").images.float().to(dev), BATCH_SIZE)
model = VAE().to(dev); opt = get_optimizer(model)


def train_graph(model, opt, x_all, bs, epochs, decay):
    """Same training as train_one_epoch x epochs (Adam step + EMA after every step, shuffled
    mini-batches), but each step is replayed from a captured CUDA graph to remove CPU overhead.
    The first epoch runs eagerly and doubles as the warm-up required before capture."""
    params = list(model.parameters())
    for g in opt.param_groups: g["capturable"] = True
    ema = [p.detach().clone() for p in params] if decay else None
    n = len(x_all)
    sizes = sorted({min(bs, n - i) for i in range(0, n, bs)})
    static_x = {k: torch.empty(k, *x_all.shape[1:], device=x_all.device) for k in sizes}
    acc = torch.zeros(3, device=x_all.device)
    model.train()

    def step(x, first=False):
        x_hat, mu, logvar = model(x)
        loss, recon, kl = loss_function(x, x_hat, mu, logvar)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if ema is not None:
            if first: torch._foreach_copy_(ema, [p.detach() for p in params])   # AveragedModel copies on 1st update
            else: torch._foreach_lerp_(ema, [p.detach() for p in params], 1 - decay)
        acc.add_(torch.stack([loss, recon, kl]).detach() * x.shape[0])

    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        xp = x_all[torch.randperm(n, device=x_all.device)]
        for j, i in enumerate(range(0, n, bs)):
            step(xp[i:i + bs], first=(j == 0))
    torch.cuda.current_stream().wait_stream(s)

    graphs, pool = {}, None
    for k in sizes:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            step(static_x[k])
        graphs[k], pool = g, g.pool()

    for e in range(1, epochs):
        if e == epochs - 1: acc.zero_()
        xp = x_all[torch.randperm(n, device=x_all.device)]
        for i in range(0, n, bs):
            k = min(bs, n - i)
            static_x[k].copy_(xp[i:i + k])
            graphs[k].replay()
    if ema is not None:
        with torch.no_grad(): torch._foreach_copy_(params, ema)
    l = (acc / n).tolist()
    return {"loss": l[0], "recon": l[1], "kl": l[2]}


use_graph = dev.type == "cuda" and os.environ.get("GRAPH", "1") == "1"
t = time.time()
if use_graph:
    decay = myoptim.EMA_DECAY if getattr(opt, "ema", None) is not None else 0
    l = train_graph(model, opt, loader.x, BATCH_SIZE, NUM_EPOCHS, decay)
else:
    for e in range(NUM_EPOCHS):
        l = train_one_epoch(model, opt, loader, dev)
torch.cuda.synchronize() if dev.type == "cuda" else None
tt = time.time() - t
r = evaluate(model)
out = dict(variant=variant, seed=seed, fid=round(r["fid"], 2), rec=round(r["reconstruction_error"], 2),
           loss=round(l["loss"], 1), recon=round(l["recon"], 1), kl=round(l["kl"], 1),
           params=sum(p.numel() for p in model.parameters()), train_s=round(tt), total_s=round(time.time() - t),
           device=torch.cuda.get_device_name() if dev.type == "cuda" else "cpu", graph=use_graph)
print(json.dumps(out), flush=True)
with open(os.path.join(HERE, "results.jsonl"), "a") as f:
    f.write(json.dumps(out) + "\n")
