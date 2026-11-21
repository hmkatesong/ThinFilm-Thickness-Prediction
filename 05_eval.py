# 5단계: 모델 예측 -> 개선 정련 -> 다중 시작점 정련 평가
#   python 05_eval.py --runs fno fno_tmm fno_synth --set valid --n 1000
#   python 05_eval.py --runs s3_resnet --set mock_refine           (방법 비교: 고정 모의고사 1만 개)
#   python 05_eval.py --runs <최종> --set test_final                (최종 모델을 정한 뒤 한 번만)
# 고정 세트는 make_eval_sets.py가 만든 _work/eval_sets.npz에서 읽는다:
#   practice(연습용, 분석·설계에 사용) / mock_refine(모의고사) / test_final(최종)
# 예전 방식(--set valid/test + --seed로 그때그때 뽑기)도 재현용으로 남겨 둔다.
import argparse, json, os, sys, time
import numpy as np
from common import (WORK, TOL, TOL_OLD, TOL_SEARCH, load_params, params_tuple, refine_lm, refine_best,
                    refine_multistart, curve_rmse)
from models import load_run
sys.stdout.reconfigure(encoding="utf-8")

FIXED = ("practice", "practice_1000", "practice_3000", "mock_refine", "test_final")
ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--set", choices=("valid", "test") + FIXED, default="mock_refine")
ap.add_argument("--n", type=int, default=0, help="0이면 세트 전체 (valid/test는 1000)")
ap.add_argument("--seed", type=int, default=5, help="valid/test에서 뽑을 때만 사용")
ap.add_argument("--starts", type=int, default=12)
ap.add_argument("--tag", default="", help="결과 파일 이름 뒤에 붙일 말 (기본: run 이름)")
ap.add_argument("--all-cands", action="store_true", help="후보 여러 개 모델이면 후보 전부를 1차 정련 출발점으로")
ap.add_argument("--extra-run", default="", help="다른 모델의 답(과 --all-cands면 후보)도 1차 정련 출발점에 추가")
ap.add_argument("--fast", action="store_true", help="모델 답 곡선이 이미 노이즈 수준이면 정련을 건너뛰는 경로도 평가")
a = ap.parse_args()
# 2026-11-20 변경: 예전에는 1차 정련(LM)을 보고용으로 한 번, refine_multistart 안에서 또 한 번 해서
# 시간이 LM만큼 더 들었다. 이제 1차 정련 결과를 넘겨 재사용한다. 정확도 결과는 같고 시간만 줄어든다.

par = params_tuple(load_params())
G = np.load(os.path.join(WORK, "grid.npy"), mmap_mode="r").reshape(810000, 226)
sp = np.load(os.path.join(WORK, "split.npz"))
if a.set in FIXED:
    flat = np.load(os.path.join(WORK, "eval_sets.npz"))[a.set]
    flat = np.sort(flat[:a.n] if a.n else flat)
else:
    pool = sp[a.set]
    if a.set == "valid":   # 04_train.py가 모델 선택에 쓴 Valid 1만 개 제외
        sel = np.random.default_rng(0).choice(sp["valid"], 10000, replace=False)
        pool = np.setdiff1d(pool, sel)
    flat = np.sort(np.random.default_rng(a.seed).choice(pool, a.n or 1000, replace=False))
X = np.asarray(G[flat], dtype=np.float64)
D = ((np.stack(np.unravel_index(flat, (30,) * 4), 1) + 1) * 10).astype(np.float64)
off = sp["off"][flat]
print(f"{a.set} {len(flat)}개 (학습 밖 층 수 1~4: {np.bincount(off, minlength=5)[1:].tolist()})", flush=True)
tot = D.sum(1)


def summary(d, r=None):
    e = np.abs(d - D)
    out = {"mae": e.mean(0).round(3).tolist(), "mae_avg": round(float(e.mean()), 3),
           "median": round(float(np.median(e)), 3), "over20": round(float((e.max(1) > 20).mean()), 4),
           "split13": round(float(np.abs((d[:, 0] - d[:, 2]) - (D[:, 0] - D[:, 2])).mean()), 3),
           "split24": round(float(np.abs((d[:, 1] - d[:, 3]) - (D[:, 1] - D[:, 3])).mean()), 3),
           "bins": {f"{lo}-{hi}": [round(float(e[(tot >= lo) & (tot < hi)].mean()), 3),
                                   round(float((e[(tot >= lo) & (tot < hi)].max(1) > 20).mean()), 4)]
                    for lo, hi in ((0, 400), (400, 600), (600, 800), (800, 10000))}}
    if r is not None:
        ok = r < TOL
        out.update(converged=round(float(ok.mean()), 4), mae_converged=round(float(e[ok].mean()), 3),
                   n_failed=int((~ok).sum()), converged_old_tol=round(float((r < TOL_OLD).mean()), 4),
                   n_wrong_but_converged=int((ok & (e.max(1) > 5)).sum()))
    return out


def show(tag, s):
    extra = (f" | 수렴 {s['converged'] * 100:5.1f}% (수렴한 것 MAE {s['mae_converged']:.3f}, 실패 {s['n_failed']}개)"
             if "converged" in s else "")
    print(f"  {tag:16s} 평균 {s['mae_avg']:7.3f} nm, 층별 {s['mae']}, 20nm 넘게 틀림 {s['over20'] * 100:4.1f}%{extra}", flush=True)


def starts_for(run, X):
    """run의 출발점 목록. 첫 번째가 그 모델의 답(P). --all-cands면 후보 여러 개 모델의 나머지 후보도 붙인다."""
    pred = load_run(os.path.join(WORK, "runs", run))
    P = pred(X)
    C = pred.last_candidates
    if not a.all_cands or C is None:
        return P, [P]
    j = np.abs(C - P[:, None]).sum(2).argmin(1)              # 고른 후보의 번호
    keep = np.ones(C.shape[:2], bool); keep[np.arange(len(X)), j] = False
    others = C[keep].reshape(len(X), C.shape[1] - 1, 4)
    return P, [P] + [others[:, m] for m in range(others.shape[1])]


results = {}
for run in a.runs:
    t0 = time.time()
    print(f"== {run}" + (f" + {a.extra_run}" if a.extra_run else "") + (" (후보 전부)" if a.all_cands else ""), flush=True)
    P, starts = starts_for(run, X)
    if a.extra_run:
        P2, s2 = starts_for(a.extra_run, X)
        starts += s2
    t_model = time.time() - t0
    r0 = curve_rmse(par, P, X)
    res = {"model": summary(P, r0), "n_starts": len(starts)}; show("모델만", res["model"])
    t1 = time.time()
    d1, r1 = refine_best(par, starts, X)                      # 1차 정련 (출발점이 여러 개면 가장 잘 맞는 것)
    t_lm = time.time() - t1
    res["lm"] = summary(d1, r1); show("+ 개선 정련", res["lm"])
    t2 = time.time()
    d2, r2 = refine_multistart(par, P, X, n_starts=a.starts, init=(d1, r1))
    t_rest = time.time() - t2
    res["multistart"] = summary(d2, r2); show("+ 다중 시작점", res["multistart"])
    if a.fast:
        # 빠른 경로: 모델 답의 곡선이 이미 노이즈 수준(TOL_SEARCH 미만)이면 정련하지 않는다.
        t3 = time.time()
        bad = np.flatnonzero(r0 >= TOL_SEARCH)
        df, rf = P.copy(), r0.copy()
        if len(bad):
            sb = [s[bad] for s in starts]
            db, rb = refine_best(par, sb, X[bad])
            db, rb = refine_multistart(par, P[bad], X[bad], n_starts=a.starts, init=(db, rb))
            df[bad], rf[bad] = db, rb
        t_fast = t_model + time.time() - t3
        res["fast"] = summary(df, rf); res["fast"]["n_refined"] = int(len(bad))
        res["fast"]["ms_per_sample"] = round(1000 * t_fast / len(flat), 1)
        show("빠른 경로", res["fast"])
        print(f"     정련한 샘플 {len(bad)}개, 샘플당 {res['fast']['ms_per_sample']:.0f} ms", flush=True)
    res["t_model_ms"] = round(1000 * t_model / len(flat), 2)
    res["t_lm_ms"] = round(1000 * t_lm / len(flat), 1)
    res["t_rest_ms"] = round(1000 * t_rest / len(flat), 1)
    for k in range(1, 5):
        m = off == k
        if m.any():
            print(f"     학습 밖 층 {k}개: n={int(m.sum())}, 최종 MAE {np.abs(d2[m] - D[m]).mean():.3f}", flush=True)
    res["time_s"] = round(t_model + t_lm + t_rest, 1)     # 빠른 경로 시험 시간은 뺌
    res["ms_per_sample"] = round(1000 * res["time_s"] / len(flat), 1)
    results[run] = res
    print(f"  ({res['time_s']:.0f}s, 샘플당 {res['ms_per_sample']:.0f} ms)", flush=True)

tag = a.tag or "-".join(a.runs)
out = os.path.join(WORK, f"eval_{a.set}_{len(flat)}_{tag}.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump({"set": a.set, "n": len(flat), "seed": a.seed, "tol": TOL, "results": results}, f,
              ensure_ascii=False, indent=1)
print("저장:", out)
