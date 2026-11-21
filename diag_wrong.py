# 진단: "곡선은 맞는 오답" (판정 기준 TOL 안인데 한 층 이상 5 nm 넘게 틀린 샘플). 연습용 세트만 사용.
#   python diag_wrong.py [--run s3_resnet] [--n-ok 40]
# 질문: 곡선만으로 구분이 안 되는 경우(찾은 답 곡선이 정답만큼 잘 맞음)인가, 탐색이 부족한 경우(정답 곡선이 더 잘 맞음)인가.
#   chi2 차이 = 226 * (r_found^2 - r_truth^2) / SIGMA^2. 크면 정답이 확실히 더 잘 맞는다 (탐색 부족).
# 시험: 오답 샘플과 정상 샘플 일부에 다시 찾기를 판정 통과 여부와 상관없이 전부 돌려 본다 (고쳐지는 수 / 망가지는 수).
import argparse, json, os, sys, time
import numpy as np
from common import (WORK, SIGMA, TOL, load_params, params_tuple, refine_lm, refine_multistart, curve_rmse,
                    jacobian)
from models import load_run
sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="s3_resnet")
ap.add_argument("--n-ok", type=int, default=40)
ap.add_argument("--n", type=int, default=0, help="시험 실행용: 연습용 세트 앞 n개만")
a = ap.parse_args()
OUT = os.path.join(WORK, "diag"); os.makedirs(OUT, exist_ok=True)
par = params_tuple(load_params())
nH, nL = float(np.median(par[0])), float(np.median(par[2]))
G = np.load(os.path.join(WORK, "grid.npy"), mmap_mode="r").reshape(810000, 226)
flat = np.sort(np.load(os.path.join(WORK, "eval_sets.npz"))["practice"])
flat = flat[:a.n] if a.n else flat
X = np.asarray(G[flat], dtype=np.float64)
D = ((np.stack(np.unravel_index(flat, (30,) * 4), 1) + 1) * 10).astype(np.float64)
t0 = time.time()
P = load_run(os.path.join(WORK, "runs", a.run))(X)
d, r = refine_multistart(par, P, X)
dt = refine_lm(par, D, X); rt = curve_rmse(par, dt, X)          # 정답에서 출발한 정련 = 정답의 곡선 오차
e = np.abs(d - D).max(1)
ok = r < TOL
wrong = np.flatnonzero(ok & (e > 5))
print(f"practice {len(flat)}개, {time.time() - t0:.0f}s. 판정 통과 {ok.sum()}, 그중 5 nm 넘게 틀림 {len(wrong)}개 "
      f"(20 nm 넘게 {int((ok & (e > 20)).sum())}개), 판정 실패 {(~ok).sum()}개", flush=True)


def crlb(dd):
    J = jacobian(par, dd)
    C = np.linalg.pinv(np.einsum("nwi,nwj->nij", J, J) / SIGMA ** 2)
    return np.sqrt(np.clip(np.einsum("nii->ni", C), 0, None))


def pattern(i):
    s13t, s13f = D[i, 0] + D[i, 2], d[i, 0] + d[i, 2]
    s24t, s24f = D[i, 1] + D[i, 3], d[i, 1] + d[i, 3]
    opt_t, opt_f = nH * s13t + nL * s24t, nH * s13f + nL * s24f
    if abs(s13t - s13f) < 5 and abs(s24t - s24f) < 5:
        return "배분(합 유지)"
    if abs(opt_t - opt_f) / opt_t < 0.02:
        return "맞바꿈(광학두께 유지)"
    return "기타"


rows = []
cr_t = crlb(D[wrong]) if len(wrong) else np.zeros((0, 4))
cr_f = crlb(d[wrong]) if len(wrong) else np.zeros((0, 4))
for k, i in enumerate(wrong):
    dchi = 226 * (r[i] ** 2 - rt[i] ** 2) / SIGMA ** 2
    rows.append({"flat": int(flat[i]), "true": D[i].tolist(), "nn": P[i].round(1).tolist(), "found": d[i].round(1).tolist(),
                 "maxerr": round(float(e[i]), 1), "r_found": round(float(r[i]), 5), "r_truth": round(float(rt[i]), 5),
                 "dchi2": round(float(dchi), 1), "kind": "구분 불가(찾은 답이 같거나 더 잘 맞음)" if dchi <= 0 else
                 ("탐색 부족(정답이 확실히 더 잘 맞음)" if dchi > 10 else "애매(차이 작음)"),
                 "pattern": pattern(i), "total": float(D[i].sum()), "crlb_true": cr_t[k].round(2).tolist(),
                 "crlb_found": cr_f[k].round(2).tolist()})
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

# 다시 찾기를 전부 돌려 보기: 오답 + 정상 샘플 일부 (모든 출발점을 다 시도, 곡선이 더 잘 맞을 때만 받아들임)
rng = np.random.default_rng(0)
good = np.flatnonzero(ok & (e <= 5))
test_ok = rng.choice(good, min(a.n_ok, len(good)), replace=False)
idx = np.concatenate([wrong, test_ok])
t1 = time.time()
force = {}
if len(idx):
    d3, r3 = refine_multistart(par, P[idx], X[idx], tol=0.0, init=(d[idx], r[idx]))   # tol=0: 모든 샘플에 모든 출발점
    e3 = np.abs(d3 - D[idx]).max(1)
    nw = len(wrong)
    force = {"n_wrong": nw, "wrong_fixed": int((e3[:nw] <= 5).sum()), "n_ok_tested": len(test_ok),
             "ok_broken": int((e3[nw:] > 5).sum()), "s_per_sample": round((time.time() - t1) / len(idx), 2)}
    print("전부 다시 찾기:", json.dumps(force, ensure_ascii=False), flush=True)
summary = {"run": a.run, "n": len(flat), "n_pass": int(ok.sum()), "n_wrong": len(wrong),
           "n_wrong_over20": int((ok & (e > 20)).sum()), "n_fail": int((~ok).sum()),
           "kinds": {k: sum(1 for x in rows if x["kind"] == k) for k in {x["kind"] for x in rows}},
           "patterns": {k: sum(1 for x in rows if x["pattern"] == k) for k in {x["pattern"] for x in rows}},
           "force_search": force, "rows": rows, "time_s": round(time.time() - t0, 1)}
with open(os.path.join(OUT, f"wrong_{a.run}.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print("저장:", os.path.join(OUT, f"wrong_{a.run}.json"))
