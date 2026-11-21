# 새 기준(TOL 0.012)에서 '수렴'으로 통과했지만 틀린 샘플 확인 (연습용 앞 200개)
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from common import WORK, TOL, load_params, params_tuple, refine_multistart, refine_lm, curve_rmse
from models import load_run
sys.stdout.reconfigure(encoding="utf-8")
flat = np.sort(np.load(os.path.join(WORK, "eval_sets.npz"))["practice"][:200])
G = np.load(os.path.join(WORK, "grid.npy"), mmap_mode="r").reshape(810000, 226)
X = np.asarray(G[flat], dtype=np.float64)
D = ((np.stack(np.unravel_index(flat, (30,) * 4), 1) + 1) * 10).astype(np.float64)
par = params_tuple(load_params())
P = load_run(os.path.join(WORK, "runs", "fno_synth"))(X)
d, r = refine_multistart(par, P, X)
dt = refine_lm(par, D, X); rt = curve_rmse(par, dt, X)
e = np.abs(d - D).max(1)
for i in np.flatnonzero(e > 5):
    print(f"true {D[i].tolist()} nn {P[i].round(1).tolist()} found {d[i].round(1).tolist()} r_found {r[i]:.4f} "
          f"r_truth_refined {rt[i]:.4f} maxerr {e[i]:.1f}")
print("r_truth_refined 분포:", np.percentile(rt, [50, 99, 100]).round(4))
