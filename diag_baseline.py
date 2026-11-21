# 진단 1-a: 출발점 비교 (Train 곡선 표에서 가장 비슷한 곡선 vs 신경망). 모의고사 정련용 1만 개.
#   python diag_baseline.py --start table|nn [--run fno_synth] [--n 0]
# 2026-11-18 결과(fno_synth, 예전 TOL 0.0105): 표 방식 최종 8.79 nm·수렴 90.2%·1.37 s/샘플,
#                                         신경망 0.353 nm·99.87%·0.20 s/샘플
import argparse, json, os, sys, time
import numpy as np
from common import WORK, TOL, load_params, params_tuple, refine_multistart
sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--start", choices=["table", "nn"], required=True)
ap.add_argument("--run", default="fno_synth")
ap.add_argument("--n", type=int, default=0)
a = ap.parse_args()
OUT = os.path.join(WORK, "diag"); os.makedirs(OUT, exist_ok=True)
par = params_tuple(load_params())
G = np.load(os.path.join(WORK, "grid.npy"), mmap_mode="r").reshape(810000, 226)
sp = np.load(os.path.join(WORK, "split.npz"))
flat = np.load(os.path.join(WORK, "eval_sets.npz"))["mock_refine"]
flat = flat[:a.n] if a.n else flat
X = np.asarray(G[flat], dtype=np.float64)
D = ((np.stack(np.unravel_index(flat, (30,) * 4), 1) + 1) * 10).astype(np.float64)
t0 = time.time()
if a.start == "table":
    tr = np.sort(sp["train"])
    T = np.asarray(G[tr], dtype=np.float32)
    Dt = ((np.stack(np.unravel_index(tr, (30,) * 4), 1) + 1) * 10).astype(np.float64)
    tn = (T.astype(np.float64) ** 2).sum(1)
    idx = np.empty(len(X), int)
    for s in range(0, len(X), 500):
        idx[s:s + 500] = (tn[None] - 2 * (X[s:s + 500].astype(np.float32) @ T.T).astype(np.float64)).argmin(1)
    P, table_mb = Dt[idx], T.nbytes / 1e6
else:
    from models import load_run
    P, table_mb = load_run(os.path.join(WORK, "runs", a.run))(X), None
t_start = time.time() - t0
t1 = time.time()
d, r = refine_multistart(par, P, X)
t_ref = time.time() - t1
e0, e, ok = np.abs(P - D), np.abs(d - D), r < TOL
res = {"start": a.start, "run": a.run, "n": len(flat), "tol": TOL,
       "start_mae_avg": round(float(e0.mean()), 3), "start_within20": round(float((e0.max(1) <= 20).mean()), 4),
       "final_mae": e.mean(0).round(3).tolist(), "final_mae_avg": round(float(e.mean()), 4),
       "final_median": round(float(np.median(e)), 4), "converged": round(float(ok.mean()), 4),
       "n_failed": int((~ok).sum()), "mae_converged": round(float(e[ok].mean()), 4),
       "t_start_ms": round(1000 * t_start / len(X), 3), "t_refine_ms": round(1000 * t_ref / len(X), 1),
       "table_mb": table_mb}
print(json.dumps(res, ensure_ascii=False), flush=True)
with open(os.path.join(OUT, f"baseline_{a.start}_{a.run if a.start == 'nn' else 'table'}.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
