"""오류 분석과 그림. 사용: python src/analyze.py E22 E24 E23

먼저 run_exp.py로 E22, E23, E24를 --eval test로 돌려 results/preds/에 예측이 있어야 한다
(preds는 git에 없다). 저장된 예측은 시드 평균 예측이라 E22는 시드 3개 앙상블 값이다.

results/error_by_unseen.csv  처음 보는 층 수 k별 MAE
results/error_by_parity.csv  k 홀수/짝수별 MAE. train은 네 층 모두 10 nm의 홀수 배라 두께 합이
                             항상 20 nm의 배수다. k가 홀수면 합이 같은 train 조합이 없다.
results/fig_error_by_unseen.png, results/fig_error_by_total.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from data import ROOT, load_train  # noqa: E402
from split import n_unseen_layers  # noqa: E402

# 한글 폰트: 윈도우, 맥, 리눅스/Colab(나눔) 순으로 설치된 것 하나를 쓴다
_HAVE = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"] if f in _HAVE),
                                   "DejaVu Sans")
plt.rcParams["axes.unicode_minus"] = False
RES = ROOT / "results"
LABELS = {"E22": "MLP + FE (시드 3개 평균)", "E24": "LightGBM + FE (시드 1개)", "E23": "외우기 기준선"}
DEFAULT = ["E22", "E24", "E23"]


def label(e):
    """R22처럼 재현 이름이면 E22 라벨을 빌려 쓰고 이름을 덧붙인다."""
    base = LABELS.get(e) or LABELS.get("E" + e[1:])
    return e if base is None else (base if e in LABELS else f"{base} [{e}]")


def load_pred(exp, ev="test"):
    d = RES / "preds"
    return np.load(d / f"{exp}_{ev}.npy"), np.load(d / f"{exp}_{ev}_idx.npy")


def main(exps):
    tag = "" if exps == DEFAULT else "_" + "_".join(exps)  # 기본이 아니면 커밋된 결과를 덮어쓰지 않게
    _, y = load_train()
    rows, errs = [], {}
    for e in exps:
        p, idx = load_pred(e)
        err = np.abs(y[idx] - p)
        errs[e] = (err, y[idx])
        k = n_unseen_layers(y[idx])
        for n in range(1, 5):
            m = k == n
            rows.append({"실험": e, "처음 보는 층 수": n, "샘플 수": int(m.sum()),
                         "MAE평균": err[m].mean(),
                         **{f"MAE_L{j+1}": err[m, j].mean() for j in range(4)}})
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / f"error_by_unseen{tag}.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    par = []
    for e in exps:
        err, yy = errs[e]
        odd = n_unseen_layers(yy) % 2 == 1
        for name, m in [("짝수(1-NN이 합 맞출 수 있음)", ~odd), ("홀수(합 같은 train 없음)", odd)]:
            par.append({"실험": e, "처음 보는 층 수": name, "샘플 수": int(m.sum()), "MAE평균": err[m].mean()})
    par = pd.DataFrame(par)
    par.to_csv(RES / f"error_by_parity{tag}.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    print(par.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # 그림 1: 처음 보는 층 수별 MAE
    fig, ax = plt.subplots(figsize=(6, 3.6))
    w = 0.8 / len(exps)
    colors = ["#c0392b", "#7f7f7f", "#bdbdbd"]
    for i, e in enumerate(exps):
        t = tab[tab["실험"] == e]
        ax.bar(t["처음 보는 층 수"] + (i - (len(exps) - 1) / 2) * w, t["MAE평균"], w,
               label=label(e), color=colors[i % 3])
    ax.set_xlabel("학습에 없던 두께인 층 수")
    ax.set_ylabel("MAE (nm)")
    ax.set_xticks([1, 2, 3, 4])
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RES / f"fig_error_by_unseen{tag}.png", dpi=200)

    # 그림 2: 총 두께별 MAE (첫 실험)
    err, yy = errs[exps[0]]
    tot = yy.sum(axis=1)
    bins = np.arange(0, 1201, 100)
    c = np.digitize(tot, bins)
    mids, vals = [], []
    for b in np.unique(c):
        m = c == b
        if m.sum() >= 200:
            mids.append(tot[m].mean())
            vals.append(err[m].mean())
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(mids, vals, "o-", color="#c0392b")
    ax.set_xlabel("네 층 두께 합 (nm)")
    ax.set_ylabel("MAE (nm)")
    ax.set_title(f"{label(exps[0])}: 총 두께별 오차 (test)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RES / f"fig_error_by_total{tag}.png", dpi=200)
    for mm, v in zip(mids, vals):
        print(f"총두께 {mm:6.0f} nm: MAE {v:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
