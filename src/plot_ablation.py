"""experiment_log.csv에서 피처 기여도 표와 그림을 만든다.

results/ablation.csv, results/fig_ablation.png
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from data import ROOT  # noqa: E402

# 한글 폰트: 윈도우, 맥, 리눅스/Colab(나눔) 순으로 설치된 것 하나를 쓴다
_HAVE = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"] if f in _HAVE),
                                   "DejaVu Sans")
plt.rcParams["axes.unicode_minus"] = False

# (실험, 그림 라벨). LightGBM, skip/valid, 시드 0, 같은 설정(lr 0.1, 5000그루)
# 가운데 5줄은 누적이 아니라 원시 226개에 그룹 하나씩만 붙인 결과다.
ROWS = [("E07", "원시 226개"), ("E09", "원시 + SG 미분"), ("E08", "원시 + FFT"), ("E13", "원시 + PCA"),
        ("E12", "원시 + 구간 통계"), ("E10", "원시 + 봉우리"), ("E16", "FE 4종만 (원시 없음)"),
        ("E14", "원시 + FE 4종")]
KNN = "E02"


def main():
    log = pd.read_csv(ROOT / "experiment_log.csv", encoding="utf-8-sig").drop_duplicates("실험", keep="last")
    log = log.set_index("실험")
    base = float(log.loc["E07", "MAE평균"])
    tab = pd.DataFrame([{"실험": e, "구성": lab, "피처수": int(log.loc[e, "피처수"]),
                         "MAE": float(log.loc[e, "MAE평균"]),
                         "원시대비(%)": 100 * (float(log.loc[e, "MAE평균"]) / base - 1)} for e, lab in ROWS])
    tab.to_csv(ROOT / "results" / "ablation.csv", index=False, encoding="utf-8-sig", float_format="%.2f")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = ["#7f7f7f"] + ["#bdbdbd"] * 5 + ["#c0392b"] * 2
    ax.barh(tab["구성"][::-1], tab["MAE"][::-1], color=colors[::-1])
    knn = float(log.loc[KNN, "MAE평균"])
    ax.axvline(knn, ls="--", color="black", lw=1)
    ax.set_ylim(-0.6, len(tab) + 0.2)
    ax.text(knn, len(tab) - 0.45, f"외우기 기준선 {knn:.1f} ", ha="right", va="bottom", fontsize=9)
    for y, v in enumerate(tab["MAE"][::-1]):
        ax.text(v + 0.3, y, f"{v:.1f}", va="center", fontsize=9)
    ax.set_xlabel("검증 MAE (nm), LightGBM, 건너뛰기 분할 (FE 4종 = SG 뺀 나머지)")
    ax.set_xlim(0, max(tab["MAE"]) * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "fig_ablation.png", dpi=200)


if __name__ == "__main__":
    main()
