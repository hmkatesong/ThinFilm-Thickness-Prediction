# 팀 결과 교차 분석 (practice 3,954개만 사용, test_final은 쓰지 않음)
#   A. 층별 이론 정밀도(CRLB): 모든 팀원 모델에서 layer_2가 가장 어려운 이유 확인
#   B. 외우기 기준선(1-NN, 윤서 브랜치)의 홀짝 현상: 1-NN이 고른 곡선이 '두께 합'이 아니라
#      '광학 두께(n*d 합)'를 맞추는지 확인
# 실행: python _work/diag/team_crosscheck.py  -> _work/diag/team_crosscheck.json
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from common import WORK, SIGMA, load_params, params_tuple, tmm, jacobian  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
t0 = time.time()
P = load_params()
par = params_tuple(P)
nH, nL = float(P[:, 0].mean()), float(P[:, 2].mean())   # SiN, SiO2 평균 굴절률

grid = np.load(os.path.join(WORK, "grid.npy"), mmap_mode="r").reshape(-1, 226)
sp = np.load(os.path.join(WORK, "split.npz"))
ev = np.load(os.path.join(WORK, "eval_sets.npz"))
tr, q = sp["train"], ev["practice"]


def thick(flat):
    return (np.stack(np.unravel_index(flat, (30,) * 4), 1) + 1) * 10.0


Xtr = np.asarray(grid[np.sort(tr)], dtype=np.float32)
Dtr = thick(np.sort(tr))
Xq = np.asarray(grid[np.sort(q)], dtype=np.float32)
Dq = thick(np.sort(q))
k = (~np.isin(Dq, np.arange(10, 300, 20))).sum(1)          # 학습에 없던 두께인 층 수
odd = k % 2 == 1


def ot(D):
    return nH * (D[:, 0] + D[:, 2]) + nL * (D[:, 1] + D[:, 3])


# ---------- A. CRLB ----------
J = jacobian(par, Dq)
C = np.linalg.pinv(np.einsum("nwi,nwj->nij", J, J) / SIGMA ** 2)
cr = np.sqrt(np.clip(np.einsum("nii->ni", C), 0, None))
tot = Dq.sum(1)
bins = [(0, 400), (400, 600), (600, 800), (800, 10000)]
A = {"crlb_median": np.median(cr, 0).round(3).tolist(),
     "crlb_mean": cr.mean(0).round(3).tolist(),
     "crlb_median_by_total": {f"{a}-{b}": np.median(cr[(tot >= a) & (tot < b)], 0).round(3).tolist()
                              for a, b in bins}}
print("A. CRLB 중앙값 L1~L4:", A["crlb_median"], flush=True)

# ---------- B. 1-NN ----------
tn = (Xtr ** 2).sum(1)
nn = np.empty(len(Xq), int)
for s in range(0, len(Xq), 500):
    xb = Xq[s:s + 500]
    d2 = (xb ** 2).sum(1)[:, None] - 2 * xb @ Xtr.T + tn[None]
    nn[s:s + 500] = d2.argmin(1)
Dn = Dtr[nn]
err_nn = np.abs(Dn - Dq)
rms_nn = np.sqrt(((Xtr[nn] - Xq) ** 2).mean(1))

# 비교 후보: 학습에 없던 층만 ±10 nm로 옮겨 가장 가까운 학습 격자점 (2^k개 중 곡선이 가장 비슷한 것)
best_d, best_r = np.empty_like(Dq), np.full(len(Dq), np.inf)
for m in range(16):
    sgn = np.array([1 if (m >> j) & 1 else -1 for j in range(4)], float)
    cand = Dq.copy()
    un = ~np.isin(Dq, np.arange(10, 300, 20))
    cand[un] = (Dq + 10 * sgn)[un]
    okc = ((cand >= 10) & (cand <= 290)).all(1)
    r = np.full(len(Dq), np.inf)
    r[okc] = np.sqrt(((tmm(par, cand[okc]) - Xq[okc]) ** 2).mean(1))
    upd = r < best_r
    best_r[upd], best_d[upd] = r[upd], cand[upd]
err_grid = np.abs(best_d - Dq)

d_sum_nn = Dn.sum(1) - Dq.sum(1)
d_ot_nn = ot(Dn) - ot(Dq)
d_ot_grid = ot(best_d) - ot(Dq)


def summ(mask):
    return {"n": int(mask.sum()),
            "nn_mae": round(float(err_nn[mask].mean()), 2),
            "nn_layer_mae": err_nn[mask].mean(0).round(2).tolist(),
            "nn_curve_rmse_median": round(float(np.median(rms_nn[mask])), 4),
            "nn_abs_dsum_median": round(float(np.median(np.abs(d_sum_nn[mask]))), 1),
            "nn_abs_dOT_median": round(float(np.median(np.abs(d_ot_nn[mask]))), 1),
            "grid_mae": round(float(err_grid[mask].mean()), 2),
            "grid_curve_rmse_median": round(float(np.median(best_r[mask])), 4),
            "grid_abs_dOT_median": round(float(np.median(np.abs(d_ot_grid[mask]))), 1),
            "frac_nn_curve_closer_than_grid": round(float((rms_nn[mask] < best_r[mask]).mean()), 3)}


B = {"all": summ(np.ones(len(Dq), bool)), "even": summ(~odd), "odd": summ(odd),
     **{f"k{j}": summ(k == j) for j in range(1, 5)}}
for key, v in B.items():
    print(f"B. {key}: {v}", flush=True)

out = {"set": "practice", "n": int(len(q)), "nH_mean": round(nH, 3), "nL_mean": round(nL, 3),
       "sigma": SIGMA, "A_crlb": A, "B_1nn": B, "time_s": round(time.time() - t0, 1)}
with open(os.path.join(HERE, "team_crosscheck.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"저장: team_crosscheck.json ({out['time_s']}s)")
