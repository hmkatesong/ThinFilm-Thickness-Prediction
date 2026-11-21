# 3-1 빠른 확인: 연습용 세트의 실패 6개(어제 분석)에 새 정련 규칙 적용
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from common import WORK, TOL, load_params, params_tuple, refine_multistart, curve_rmse
from models import load_run
sys.stdout.reconfigure(encoding="utf-8")
flat = np.array([164250, 241290, 250009, 423032, 464695, 746064])
G = np.load(os.path.join(WORK, "grid.npy"), mmap_mode="r").reshape(810000, 226)
X = np.asarray(G[flat], dtype=np.float64)
D = ((np.stack(np.unravel_index(flat, (30,) * 4), 1) + 1) * 10).astype(np.float64)
par = params_tuple(load_params())
P = load_run(os.path.join(WORK, "runs", "fno_synth"))(X)
for tag, kw in (("old(0.0105, no swap)", dict(tol=0.0105, swap=False)), ("new", {})):
    d, r = refine_multistart(par, P, X, **kw)
    print(tag)
    for i in range(len(flat)):
        print(f"  {D[i].tolist()} -> {d[i].round(1).tolist()}  r={r[i]:.4f}  maxerr={np.abs(d[i] - D[i]).max():.2f}  ok={r[i] < (0.0105 if 'tol' in kw else TOL)}")
