"""Mean and range of FID / reconstruction error per variant, from colab/results.jsonl."""
import os, json, collections, statistics as S
d = collections.defaultdict(list)
for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl")):
    r = json.loads(line); d[r["variant"]].append(r)
print(f"{'variant':26s} n   fid (min-max)        rec (min-max)")
for v, rs in sorted(d.items(), key=lambda kv: S.mean(r["fid"] for r in kv[1])):
    f = [r["fid"] for r in rs]; c = [r["rec"] for r in rs]
    print(f"{v:26s} {len(rs)}  {S.mean(f):6.2f} ({min(f):5.1f}-{max(f):5.1f})  {S.mean(c):6.2f} ({min(c):5.1f}-{max(c):5.1f})")
