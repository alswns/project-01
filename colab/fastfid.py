"""Same numbers as util/result.evaluate, but caches the real-image Inception stats."""
import os, torch
from torcheval.metrics import FrechetInceptionDistance
from data import get_dataset
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_state.pt")
_fid = None
def rgb(x): return x.float().clamp(0, 1).repeat(1, 3, 1, 1)
@torch.no_grad()
def evaluate(model, bs=None):
    bs = bs or int(os.environ.get("FID_BS", 50))
    n = int(os.environ.get("FID_N", 5000))   # generated images for the FID (5000 = official)
    global _fid
    dev = next(model.parameters()).device
    test = get_dataset("test").images
    model.eval()
    if _fid is None:
        _fid = FrechetInceptionDistance(device=dev)
        if os.path.exists(CACHE):
            st = torch.load(CACHE)
            for k in ("real_sum", "real_cov_sum", "num_real_images"): setattr(_fid, k, st[k].to(dev))
        else:
            for x in test.split(bs): _fid.update(rgb(x.to(dev)), is_real=True)
            torch.save({k: getattr(_fid, k).cpu() for k in ("real_sum", "real_cov_sum", "num_real_images")}, CACHE)
        _real = {k: getattr(_fid, k).clone() for k in ("real_sum", "real_cov_sum", "num_real_images")}
        _fid._real = _real
    for k, v in _fid._real.items(): setattr(_fid, k, v.clone())
    for k in ("fake_sum", "fake_cov_sum"): setattr(_fid, k, torch.zeros_like(getattr(_fid, k)))
    _fid.num_fake_images = torch.tensor(0, device=dev).int()
    z = torch.randn(len(test), model.latent_dim, generator=torch.Generator().manual_seed(0))[:n]
    se = 0.0
    for x in test.split(500):
        x = x.to(dev); mu, _ = model.encode(x); se += (x - model.decode(mu)).pow(2).sum().item()
    for zb in z.split(bs):
        _fid.update(rgb(model.decode(zb.to(dev))), is_real=False)
    return {"fid": _fid.compute().item(), "reconstruction_error": se / len(test)}
