# 3단계: Train(20 nm 부분격자)만으로 파장별 광학 상수 적합 (부록 A.3 --subgrid 기반)
# 출력: _work/optical_params.csv  (w,nH,kH,nL,kL,ns,ks,lam_nm,holdout_rmse)
# 홀드아웃은 Valid에서만 뽑는다(라벨은 평가에만 사용, Test는 건드리지 않음).
import os, sys, time
import numpy as np
from scipy.optimize import least_squares
from common import WORK, HC, tmm
sys.stdout.reconfigure(encoding="utf-8")

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--n-fit", type=int, default=1500, help="적합에 쓸 Train 샘플 수 (최대 50625)")
ap.add_argument("--out", default="optical_params.csv")
args = ap.parse_args()
N_FIT, N_HOLD = args.n_fit, 2000
OUT = os.path.join(WORK, args.out)
G = np.load(os.path.join(WORK, "grid.npy"), mmap_mode="r")
sp = np.load(os.path.join(WORK, "split.npz"))
rng = np.random.default_rng(7)
fit_flat = rng.choice(sp["train"], N_FIT, replace=False)
hold_flat = rng.choice(sp["valid"], N_HOLD, replace=False)
np.save(os.path.splitext(OUT)[0] + "_fit_idx.npy", fit_flat)


def load(flat):
    i = np.stack(np.unravel_index(flat, (30,) * 4), 1)
    return np.stack([np.asarray(G[tuple(r)]) for r in i]).astype(np.float64), (i + 1) * 10.0


Ytr, Dtr = load(fit_flat)
Yte, Dte = load(hold_flat)

LO = np.array([1.0, 0, 1.0, 0, 0.5, 0, 0])
HI = np.array([4.0, 2, 3.0, 2, 9.0, 9, 2000])
SC = np.array([1, .01, 1, .01, 1, .1, 100])
SUB = [(3.7, .01), (4.2, .03), (5.0, .1), (5.6, .4), (6.3, 2.6), (5.1, 3.1), (5.0, 3.8), (3.6, 5.2)]


def base(w):
    E = 1.55 + 0.0124 * w
    return np.array([2.0 + 0.04 * (E - 1.55), 1e-3, 1.47 + 0.012 * (E - 1.55), 1e-3, 4.0, 0.1, HC / E])


def solve(w, x0):
    x0 = np.clip(x0, LO + 1e-6, HI - 1e-6)
    return least_squares(lambda p: tmm(p, Dtr) - Ytr[:, w], x0, bounds=(LO, HI), x_scale=SC, max_nfev=40)


P = np.zeros((226, 7)); te = np.zeros(226); t0 = time.time()
for w in range(226):
    starts = [P[w - 1]] if w else []
    if w % 25 == 0:
        for ns, ks in SUB:
            s = base(w); s[4], s[5] = ns, ks; starts.append(s)
    else:
        s = base(w); s[4], s[5] = P[w - 1][4], P[w - 1][5]; starts.append(s)
    best = min((solve(w, s) for s in starts), key=lambda r: r.cost)
    P[w] = best.x
    te[w] = np.sqrt(((tmm(best.x, Dte) - Yte[:, w]) ** 2).mean())
    if w % 25 == 0:
        print(f"w={w} holdout={te[w]:.4f} {time.time() - t0:.0f}s", flush=True)

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write("w,nH,kH,nL,kL,ns,ks,lam_nm,holdout_rmse\n")
    for w in range(226):
        p = P[w]
        f.write(f"{w},{p[0]:.5f},{p[1]:.6f},{p[2]:.5f},{p[3]:.6f},{p[4]:.5f},{p[5]:.5f},{p[6]:.3f},{te[w]:.5f}\n")
E = HC / P[:, 6]
a, b = np.polyfit(np.arange(226), E, 1)
print(f"에너지 축: E = {b:.3f} + {a:.5f}*w, 범위 {P[:, 6].min():.0f}~{P[:, 6].max():.0f} nm")
print("Valid 홀드아웃 RMSE", round(float(np.sqrt((te ** 2).mean())), 5),
      "파장별 최소/중앙/최대", te.min().round(5), np.median(te).round(5), te.max().round(5))
