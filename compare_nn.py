# 신경망만으로 비교 (정련 없음). 모의고사 신경망용 세트(mock_nn, 51,712개).
#   python compare_nn.py s2_resnet s3_resnet s3_resnet_k4 --out compare_s3
import argparse, json, os, sys, time
import numpy as np
from common import WORK
from models import load_run
sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", default="compare_nn")
a = ap.parse_args()
OUT = os.path.join(WORK, "diag"); os.makedirs(OUT, exist_ok=True)
G = np.load(os.path.join(WORK, "grid.npy"), mmap_mode="r").reshape(810000, 226)
flat = np.load(os.path.join(WORK, "eval_sets.npz"))["mock_nn"]
X = np.asarray(G[flat], dtype=np.float64)
D = ((np.stack(np.unravel_index(flat, (30,) * 4), 1) + 1) * 10).astype(np.float64)
tot = D.sum(1)
res = {}
for run in a.runs:
    rd = os.path.join(WORK, "runs", run)
    if not os.path.exists(os.path.join(rd, "best.pt")):
        print(run, "best.pt 없음", flush=True); continue
    pred = load_run(rd)
    t = time.time(); P = pred(X); dt = time.time() - t
    e = np.abs(P - D)
    o = {"over20": round(float((e.max(1) > 20).mean()), 4), "mae_avg": round(float(e.mean()), 3),
         "mae": e.mean(0).round(2).tolist(), "median": round(float(np.median(e)), 2),
         "split13": round(float(np.abs((P[:, 0] - P[:, 2]) - (D[:, 0] - D[:, 2])).mean()), 2),
         "sum13": round(float(np.abs((P[:, 0] + P[:, 2]) - (D[:, 0] + D[:, 2])).mean()), 2),
         "split24": round(float(np.abs((P[:, 1] - P[:, 3]) - (D[:, 1] - D[:, 3])).mean()), 2),
         "sum24": round(float(np.abs((P[:, 1] + P[:, 3]) - (D[:, 1] + D[:, 3])).mean()), 2),
         "ms_per_1000": round(1000 * dt / len(X) * 1000, 1)}
    if getattr(pred, "last_candidates", None) is not None:      # 후보 여러 개: 그중 가장 가까운 것 (상한)
        ec = np.abs(pred.last_candidates - D[:, None]).max(2).min(1)
        o["oracle_over20"] = round(float((ec > 20).mean()), 4)
    log = json.load(open(os.path.join(rd, "log.json"), encoding="utf-8"))
    curve = [round(l["valid"].get("over20", l["valid"]["mae_avg"]), 4) for l in log]
    o.update(epochs=len(log), train_s=log[-1]["time_s"], best_epoch=int(np.argmin(curve)) + 1, valid_curve=curve)
    o["bins"] = {}
    for lo, hi in [(0, 400), (400, 600), (600, 800), (800, 10000)]:
        m = (tot >= lo) & (tot < hi)
        o["bins"][f"{lo}-{hi}"] = {"n": int(m.sum()), "mae": round(float(e[m].mean()), 2),
                                    "over20": round(float((e[m].max(1) > 20).mean()), 4)}
    res[run] = o
    print(run, json.dumps(o, ensure_ascii=False), flush=True)
with open(os.path.join(OUT, f"{a.out}.json"), "w", encoding="utf-8") as f:
    json.dump({"n": len(flat), "results": res}, f, ensure_ascii=False, indent=1)
print("done", flush=True)
