# 진단: 정답 두께로 그린 TMM 곡선과 실제 곡선의 차이(잔차) 분석. 학습 없음.
#   python diag_noise.py --quick   -> 실패 판정 기준(TOL) 정하기용 분포만 (Train 정답 곡선 오차)
#   python diag_noise.py           -> 위 + 파장별 평균 잔차, 이웃 파장 상관, 평균 잔차를 빼면 정련 천장이 내려가는지
# Train만으로 통계를 내고, 보정 효과 확인은 연습용 세트에서만 한다.
import argparse, json, os, sys, time
import numpy as np
from common import WORK, SIGMA, load_params, params_tuple, tmm, refine_lm, jacobian
sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
OUT = os.path.join(WORK, "diag"); os.makedirs(OUT, exist_ok=True)
par = params_tuple(load_params())
G = np.load(os.path.join(WORK, "grid.npy"), mmap_mode="r").reshape(810000, 226)
sp = np.load(os.path.join(WORK, "split.npz"))


def lab(flat):
    return ((np.stack(np.unravel_index(flat, (30,) * 4), 1) + 1) * 10).astype(np.float64)


tr = np.sort(sp["train"])
Xt, Dt = np.asarray(G[tr], dtype=np.float64), lab(tr)
res = Xt - tmm(par, Dt)                                   # (50625, 226) 잔차
rm = np.sqrt((res ** 2).mean(1))
q = {f"p{p}": round(float(np.percentile(rm, p)), 5) for p in (50, 90, 99, 99.9, 99.99)}
q.update(mean=round(float(rm.mean()), 5), max=round(float(rm.max()), 5), n=len(rm))
print("Train 정답 곡선 오차(RMSE) 분포:", json.dumps(q), flush=True)
out = {"train_truth_rmse": q}
if a.quick:
    sys.exit()

# 1) 파장별 평균 잔차: 순수 노이즈면 0 근처 (표준오차 = sd/sqrt(n))
mu = res.mean(0); se = res.std(0) / np.sqrt(len(res))
tot = Dt.sum(1)
out["mean_resid"] = {"rms_of_mean": round(float(np.sqrt((mu ** 2).mean())), 5),
                     "max_abs_mean": round(float(np.abs(mu).max()), 5),
                     "n_sig_3se": int((np.abs(mu) > 3 * se).sum()), "of": 226, "sigma": SIGMA}
bins = [(0, 400), (400, 600), (600, 800), (800, 10000)]
out["by_total"] = []
for lo, hi in bins:
    m = (tot >= lo) & (tot < hi)
    mb = res[m].mean(0)
    out["by_total"].append({"bin": f"{lo}-{hi}", "n": int(m.sum()), "rmse": round(float(np.sqrt((res[m] ** 2).mean())), 5),
                            "rms_of_mean": round(float(np.sqrt((mb ** 2).mean())), 5)})
# 2) 이웃 파장끼리 상관 (샘플 안에서 평균을 뺀 잔차)
rc = res - res.mean(1, keepdims=True)
den = (rc ** 2).mean()
out["autocorr"] = {f"lag{k}": round(float((rc[:, :-k] * rc[:, k:]).mean() / den), 4) for k in (1, 2, 3, 5, 10)}
# 잔차를 파장마다 표준화한 뒤에도 남는 상관
rz = (res - mu) / res.std(0)
out["autocorr_std"] = {f"lag{k}": round(float((rz[:, :-k] * rz[:, k:]).mean()), 4) for k in (1, 2, 3)}
print(json.dumps({k: out[k] for k in ("mean_resid", "by_total", "autocorr", "autocorr_std")}, ensure_ascii=False), flush=True)

# 3) 보정 시험: Train 평균 잔차(파장별, 총두께 구간별)를 실제 곡선에서 빼고 정답 출발 정련 -> 오차가 줄어드나
es = np.load(os.path.join(WORK, "eval_sets.npz"))
fp = es["practice"]
Xp, Dp = np.asarray(G[fp], dtype=np.float64), lab(fp)
tp = Dp.sum(1)
mu_bin = np.zeros_like(Xp)
for lo, hi in bins:
    mt, mp = (tot >= lo) & (tot < hi), (tp >= lo) & (tp < hi)
    mu_bin[mp] = res[mt].mean(0)
J = jacobian(par, Dp)
crlb = np.sqrt(np.clip(np.einsum("nii->ni", np.linalg.pinv(np.einsum("nwi,nwj->nij", J, J) / SIGMA ** 2)), 0, None))
rows = {}
for tag, Xc in (("raw", Xp), ("minus_mean", Xp - mu), ("minus_mean_by_total", Xp - mu_bin)):
    t = time.time()
    e = np.abs(refine_lm(par, Dp, Xc) - Dp)
    rows[tag] = {"mae": round(float(e.mean()), 4), "ratio_med": round(float(np.median(e / crlb)) / 0.674, 2),
                 "bins": {f"{lo}-{hi}": [round(float(e[(tp >= lo) & (tp < hi)].mean()), 3),
                                         round(float(np.median(e[(tp >= lo) & (tp < hi)] / crlb[(tp >= lo) & (tp < hi)])) / 0.674, 2)]
                          for lo, hi in bins}, "time_s": round(time.time() - t, 1)}
    print(tag, json.dumps(rows[tag]), flush=True)
out["truth_start_refine_practice"] = rows
out["note"] = "ratio = 중앙값(|오차|/CRLB)/0.674. 1.0이면 이론 한계와 같음. bins 값은 [MAE, ratio]"
with open(os.path.join(OUT, "noise.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("저장:", os.path.join(OUT, "noise.json"))
