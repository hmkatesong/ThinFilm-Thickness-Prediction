# 발표용 그래프 만들기 -> figures/*.png
#   python make_figures.py
# 동훈 결과는 _work의 평가 json·로그에서 읽는다. 팀원 숫자는 각 브랜치 README·실험 로그 값을 옮겨 적었다(출처는 주석).
# 예시 곡선(그림 9)과 교차 분석(그림 3·5)은 practice 세트만 쓴다.
import json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import WORK, SIGMA, load_params, params_tuple, tmm, refine_lm  # noqa: E402

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")

# ---------- 스타일 (색: dataviz 기본 팔레트, 색맹 검사 통과 순서) ----------
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_L, BLUE_D, GRAY, GRAY_D = "#86b6ef", "#184f95", "#a9a8a3", "#6f6e69"
MEMBER = {"기준": GRAY, "은수": AQUA, "윤서": ORANGE, "동훈": BLUE}

fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.family": next((f for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"] if f in fonts), "DejaVu Sans"),
    "axes.unicode_minus": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.frameon": False, "legend.fontsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2, "lines.markersize": 8,
})
MK = dict(markeredgecolor=SURF, markeredgewidth=1.5, zorder=3)


def head(ax, title, sub=None, size=15):
    """제목과 부제목을 축 위에 겹치지 않게 놓는다 (포인트 단위 간격)."""
    ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(0, 32 if sub else 12), textcoords="offset points",
                fontsize=size, fontweight="bold", ha="left", va="bottom")
    if sub:
        ax.annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, 10), textcoords="offset points",
                    fontsize=10.5, color=INK2, ha="left", va="bottom")


def save(fig, name, note=None):
    if note:
        fig.text(0.01, -0.01, note, fontsize=8.5, color=MUTED, ha="left", va="top")
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("저장:", name)


def spread(vals, gap):
    """라벨 위치가 gap보다 가까우면 위로 밀어 겹치지 않게 한다."""
    order = np.argsort(vals)
    pos = np.array(vals, float)
    for a, b in zip(order[:-1], order[1:]):
        pos[b] = max(pos[b], pos[a] + gap)
    return pos


def grid_y(ax):
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


def res(path):
    d = json.load(open(os.path.join(WORK, path), encoding="utf-8"))
    return list(d["results"].values())[0]


def unseen_from_log(key):
    """05_eval.py는 학습 밖 층 수별 최종 MAE를 로그에만 남긴다."""
    txt = open(os.path.join(WORK, "runs", f"eval_tf_{key}.log"), encoding="utf-8").read()
    m = re.findall(r"학습 밖 층 (\d)개: n=\d+, 최종 MAE ([\d.]+)", txt)
    return [float(v) for _, v in m]


KEYS = ("a_plus_b", "a_base", "s4_corr")
TF = {k: res(f"eval_test_final_20000_tf_{k}.json") for k in KEYS}
MOCK = {"a_plus_b": res("eval_mock_refine_10000_s4_a_plus_b.json"),
        "a_base": res("eval_mock_refine_10000_s4_a_base.json"),
        "s4_corr": res("eval_mock_refine_10000_s4_corr.json")}
XC = json.load(open(os.path.join(WORK, "diag", "team_crosscheck.json"), encoding="utf-8"))
NAMES = {"a_plus_b": "ResNet A + 후보 B (최종)", "a_base": "ResNet A만", "s4_corr": "보정 신경망"}
STYLE = {"a_plus_b": (BLUE, "o", "-"), "a_base": (BLUE_L, "s", "-"), "s4_corr": (BLUE_D, "D", "--")}
fin = TF["a_plus_b"]["multistart"]["mae_avg"]

# ============ 그림 1. 팀 결과 사다리 ============
rows = [  # (이름, MAE, 담당, 평가 세트)
    ("평균값 예측", 72.31, "기준", "valid"),                         # 은수 README
    ("Ridge (원시 226개)", 72.52, "윤서", "valid"),                  # 윤서 experiment_log E03
    ("Ridge + 도메인 피처", 54.02, "은수", "valid"),                 # 은수 README
    ("LightGBM (원시)", 35.38, "윤서", "test"),                      # 윤서 E26
    ("LightGBM + 피처", 26.92, "윤서", "test"),                      # 윤서 E24
    ("외우기 기준선 (1-NN)", 26.44, "윤서", "test"),                 # 윤서 E23
    ("MLP + 피처 (시드 3개 평균)", 16.94, "윤서", "test"),           # 윤서 E22
    ("1D ResNet + TMM 합성 데이터", TF["a_plus_b"]["model"]["mae_avg"], "동훈", "test_final"),
    ("+ 보정 신경망", TF["s4_corr"]["model"]["mae_avg"], "동훈", "test_final"),
    ("+ TMM 물리 피팅 (최종)", fin, "동훈", "test_final"),
]
fig, ax = plt.subplots(figsize=(11, 6.2))
y = np.arange(len(rows))[::-1]
for yi, (n, v, who, st) in zip(y, rows):
    ax.barh(yi, v, height=0.62, color=MEMBER[who], edgecolor=SURF, linewidth=2)
    bold = who == "동훈" and v < 1
    ax.annotate(f"{v:.3g} nm" if v < 1 else f"{v:.1f} nm", (v, yi), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=11 if bold else 10.5, fontweight="bold" if bold else "normal")
    ax.annotate(st, (v, yi), xytext=(66, 0), textcoords="offset points", va="center", fontsize=8.5, color=MUTED)
ax.set_yticks(y, [r[0] for r in rows])
ax.set_xscale("log")
ax.set_xlim(0.1, 500)
ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
ax.set_xlabel("평균 절대 오차 MAE (nm, 로그 축)")
ax.xaxis.grid(True)
ax.set_axisbelow(True)
ax.legend(handles=[Patch(color=MEMBER[k], label=k) for k in ("기준", "은수", "윤서", "동훈")],
          loc="lower right", title="담당", title_fontsize=10, alignment="left")
head(ax, "반사율 → 두께: 팀 전체 결과",
     f"평균값 예측 72 nm에서 TMM 물리 피팅 {fin:.3g} nm까지, 약 {72.31 / fin:.0f}분의 1")
save(fig, "fig1_team_ladder.png",
     "건너뛰기 분할(학습 밖 두께). 평가 세트가 팀원마다 달라 대략 비교. 은수·윤서 값은 각 브랜치 README·실험 로그, "
     "동훈 값은 test_final 2만 개.")

# ============ 그림 2. 단계별 오차 (test_final) ============
fig, ax = plt.subplots(figsize=(11, 5.8))
x = np.arange(3)
finals = []
for k in KEYS:
    c, mk, ls = STYLE[k]
    r = TF[k]
    v = [r["model"]["mae_avg"], r["lm"]["mae_avg"], r["multistart"]["mae_avg"]]
    finals.append(v[2])
    ax.plot(x, v, color=c, marker=mk, ls=ls, **MK)
    ax.annotate(f"{v[1]:.2f}", (1, v[1]), xytext=(10, 0), textcoords="offset points", va="center",
                fontsize=9.5, color=INK2)
ax.annotate("5.12  (A, A+B 공통)", (0, TF["a_base"]["model"]["mae_avg"]), xytext=(-10, 0),
            textcoords="offset points", ha="right", va="center", fontsize=9.5, color=INK2)
ax.annotate(f"{TF['s4_corr']['model']['mae_avg']:.2f}", (0, TF["s4_corr"]["model"]["mae_avg"]), xytext=(-10, 0),
            textcoords="offset points", ha="right", va="center", fontsize=9.5, color=INK2)
lab = 10 ** spread(np.log10(finals), 0.09)
for k, v, ly in zip(KEYS, finals, lab):
    ms = TF[k]["ms_per_sample"]
    ax.annotate(f"{NAMES[k]}   {v:.3f} nm · {ms:.0f} ms/샘플", (2, v), xytext=(2.12, ly), textcoords="data",
                va="center", fontsize=10.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=0, shrinkB=4))
ax.set_yscale("log")
ax.set_ylim(0.2, 10)
ax.set_yticks([0.2, 0.5, 1, 2, 5, 10], ["0.2", "0.5", "1", "2", "5", "10"])
ax.set_xticks(x, ["신경망만", "+ LM 정련 1회", "+ 다시 찾기 (최종)"])
ax.set_xlim(-0.75, 3.9)
ax.set_ylabel("MAE (nm, 로그 축)")
grid_y(ax)
head(ax, "최종 정확도는 물리 피팅이 정한다",
     "신경망 오차가 1.3~5.1 nm로 달라도 TMM 피팅 후에는 0.28~0.31 nm로 모인다 (test_final 2만 개)")
save(fig, "fig2_pipeline_stages.png",
     "후보 B를 함께 출발점으로 쓰면 정련 1회 후 오차가 1.08 → 0.47 nm로 줄어 다시 찾기 부담이 준다. 시간은 CPU 기준.")

# ============ 그림 3. 층별 난이도: L2/L1 비율 ============
cr = XC["A_crlb"]["crlb_median"]
tfl, nnl = TF["a_plus_b"]["multistart"]["mae"], TF["a_plus_b"]["model"]["mae"]
lay = [  # (이름, L1, L2, 담당)
    ("외우기 1-NN (윤서, test)", 18.24, 34.79, "윤서"),           # E23
    ("LightGBM + 피처 (은수, valid)", 24.44, 42.40, "은수"),      # 은수 README
    ("LightGBM + 피처 (윤서, test)", 20.93, 34.94, "윤서"),       # E24
    ("MLP + 피처 (윤서, test)", 17.08, 26.69, "윤서"),            # E22
    ("ResNet 신경망만 (동훈, test_final)", nnl[0], nnl[1], "동훈"),
    ("TMM 피팅 최종 (동훈, test_final)", tfl[0], tfl[1], "동훈"),
]
fig, ax = plt.subplots(figsize=(10.5, 5.6))
y = np.arange(len(lay))[::-1]
for yi, (n, a, b, who) in zip(y, lay):
    ax.barh(yi, b / a, height=0.6, color=MEMBER[who], edgecolor=SURF, linewidth=2)
    ax.annotate(f"{b / a:.2f}", (b / a, yi), xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                fontsize=11, color="white", fontweight="bold")
ref = cr[1] / cr[0]
ax.axvline(ref, color=INK, ls="--", lw=1.5)
ax.annotate(f"이론 한계(CRLB) 비율 {ref:.2f}", (ref, -0.75), xytext=(6, 0), textcoords="offset points",
            fontsize=10, va="center", color=INK)
ax.set_ylim(-1.1, len(lay) - 0.5)
ax.set_yticks(y, [r[0] for r in lay])
ax.set_xlim(1.0, 2.1)
ax.set_xlabel("layer_2 오차 ÷ layer_1 오차")
ax.xaxis.grid(True)
ax.set_axisbelow(True)
head(ax, "모든 모델에서 layer_2(가운데 SiO₂)가 가장 어렵다",
     "오차 크기는 100배 차이 나도 L2/L1 비율은 1.6~1.9로, 데이터에 담긴 정보량의 한계(1.81)와 같다")
save(fig, "fig3_layer2_difficulty.png",
     "CRLB: 노이즈 σ=0.0087에서 어떤 방법도 넘을 수 없는 두께 정밀도(practice 3,954개 중앙값). "
     "선형 회귀(hmsong)는 아무것도 못 배워 비율 1.00.")

# ============ 그림 4. 총두께별 난이도 ============
bins = ["0-400", "400-600", "600-800", "800-10000"]
labels = ["400 미만", "400~600", "600~800", "800 이상"]
crb = np.array([np.mean(XC["A_crlb"]["crlb_median_by_total"][b]) for b in bins])
nn = np.array([TF["a_plus_b"]["model"]["bins"][b][0] for b in bins])
fa = np.array([TF["a_base"]["multistart"]["bins"][b][0] for b in bins])
ff = np.array([TF["a_plus_b"]["multistart"]["bins"][b][0] for b in bins])
series = [("ResNet 신경망만", nn, BLUE_L, "s", "-"), ("A만 + TMM 피팅", fa, BLUE_D, "D", "-"),
          ("최종 (A + 후보 B + TMM 피팅)", ff, BLUE, "o", "-"), ("이론 한계 (CRLB)", crb, INK2, "^", "--")]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.2, 1]})
x = np.arange(4)
ratios = [v[0] / v[-1] for _, v, *_ in series]
ly = spread(ratios, 0.32)
for (n, v, c, mk, ls), r0, yy in zip(series, ratios, ly):
    ax.plot(x, v / v[-1], color=c, marker=mk, ls=ls, **MK)
    ax.annotate(f"{n}  ×{r0:.1f}", (0, r0), xytext=(-0.12, yy), textcoords="data", ha="right", va="center",
                fontsize=10, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=0, shrinkB=4))
ax.set_xticks(x, labels)
ax.set_xlim(-2.5, 3.25)
ax.set_ylim(0.8, 5.2)
ax.spines["bottom"].set_bounds(-0.2, 3.2)
ax.set_xlabel("네 층 두께 합 (nm)")
ax.set_ylabel("오차 배수 (800 nm 이상 = 1)")
grid_y(ax)
head(ax, "얇은 스택일수록 어렵다", "이론상 1.5배인데 모델은 3~5배: 얇은 쪽에 개선 여지가 남아 있다")
for (n, v, c, mk, ls), short in zip(series[1:], ["A만", "최종", "이론 한계"]):
    ax2.plot(x, v, color=c, marker=mk, ls=ls, **MK)
    ax2.annotate(f"{short} {v[0]:.3f}", (0, v[0]), xytext=(-10, 0), textcoords="offset points", ha="right",
                 va="center", fontsize=10)
ax2.annotate(f"{ff[-1]:.3f}", (3, ff[-1]), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=10)
ax2.set_xticks(x, labels)
ax2.set_xlim(-1.1, 3.3)
ax2.spines["bottom"].set_bounds(-0.2, 3.2)
ax2.set_xlabel("네 층 두께 합 (nm)")
ax2.set_ylabel("MAE (nm)")
ax2.set_ylim(0, 0.8)
grid_y(ax2)
head(ax2, "TMM 피팅 후 절대 오차", f"후보 B는 얇은 스택에서만 효과 ({fa[0]:.3f} → {ff[0]:.3f} nm)")
fig.tight_layout(w_pad=4)
save(fig, "fig4_total_thickness.png",
     "test_final 2만 개. CRLB는 practice 3,954개의 네 층 평균(표준편차 기준, 정규분포면 MAE는 CRLB의 약 0.8배).")

# ============ 그림 5. 외우기 홀짝 = 광학 두께 ============
B = XC["B_1nn"]
fig, axs = plt.subplots(1, 3, figsize=(13.5, 5.2))
panels = [
    ("두께 오차 MAE (nm)", "nn_mae", "grid_mae", "① 두께로는 옆 격자점이 훨씬 가깝다", "{:.1f}"),
    ("광학 두께 차이 중앙값 (nm)", "nn_abs_dOT_median", "grid_abs_dOT_median", "② 광학 두께는 1-NN이 더 잘 맞춘다", "{:.1f}"),
    ("곡선 차이 RMSE 중앙값", "nn_curve_rmse_median", "grid_curve_rmse_median", "③ 그래서 곡선은 1-NN이 더 비슷하다", "{:.3f}"),
]
w = 0.36
for ax, (yl, k1, k2, t, fmt) in zip(axs, panels):
    xx = np.arange(2)
    v1 = [B["even"][k1], B["odd"][k1]]
    v2 = [B["even"][k2], B["odd"][k2]]
    for off, vals, c in ((-w / 2 - 0.01, v1, ORANGE), (w / 2 + 0.01, v2, GRAY)):
        bars = ax.bar(xx + off, vals, w, color=c)
        for bb, vv in zip(bars, vals):
            ax.annotate(fmt.format(vv), (bb.get_x() + bb.get_width() / 2, vv), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xx, ["짝수", "홀수"])
    ax.set_xlabel("학습에 없던 층 수")
    ax.set_ylabel(yl)
    ax.set_ylim(0, max(v1 + v2) * 1.15)
    grid_y(ax)
    head(ax, t, size=12)
    if k1.startswith("nn_curve"):
        ax.axhline(SIGMA, color=INK2, ls=":", lw=1.5)
        ax.annotate("노이즈 σ", (0.5, SIGMA), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=INK2)
fig.legend(handles=[Patch(color=ORANGE, label="외우기 1-NN이 고른 학습 곡선"),
                    Patch(color=GRAY, label="바로 옆 학습 격자점 (학습 밖 층만 ±10 nm)")],
           loc="upper left", bbox_to_anchor=(0.01, 0.905), ncol=2)
fig.suptitle("외우기 기준선의 홀짝 현상: 반사율 곡선은 '광학 두께'가 정한다", x=0.01, y=0.99, ha="left",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.83), w_pad=3)
save(fig, "fig5_parity_optical_thickness.png",
     "practice 3,954개. 광학 두께 = 2.074×(L1+L3) + 1.483×(L2+L4). 홀수일 때 옆 격자점은 광학 두께가 15 nm 달라져 "
     "곡선이 크게 바뀌고, 1-NN은 두께가 35 nm 떨어져도 광학 두께가 맞는 곡선을 고른다.")

# ============ 그림 6. 학습 밖 층 수별 오차 ============
k = [1, 2, 3, 4]
team = [("외우기 1-NN", [36.21, 16.48, 31.30, 25.21], GRAY, "s", "--"),    # 윤서 results/error_by_unseen.csv E23
        ("LightGBM + 피처", [27.80, 25.73, 27.82, 26.92], GRAY_D, "D", "-"),  # E24
        ("MLP + 피처", [16.46, 16.46, 17.93, 18.30], ORANGE, "o", "-")]     # E22
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
ly = spread([v[-1] for _, v, *_ in team], 2.2)
for (n, v, c, mk, ls), yy in zip(team, ly):
    ax.plot(k, v, color=c, marker=mk, ls=ls, **MK)
    ax.annotate(n, (4, v[-1]), xytext=(4.15, yy), textcoords="data", va="center", fontsize=10)
ax.set_xticks(k)
ax.set_xlim(0.7, 5.4)
ax.set_ylim(0, 40)
ax.set_xlabel("학습에 없던 두께인 층 수")
ax.set_ylabel("MAE (nm)")
grid_y(ax)
head(ax, "윤서 브랜치 (test 10만 개)", "외우기는 홀수에서 크게 틀리고, MLP는 고르다")
off = {key: unseen_from_log(key) for key in KEYS}
ly = spread([off[key][-1] for key in KEYS], 0.02)
for key, yy in zip(KEYS, ly):
    c, mk, ls = STYLE[key]
    ax2.plot(k, off[key], color=c, marker=mk, ls=ls, **MK)
    ax2.annotate(NAMES[key], (4, off[key][-1]), xytext=(4.15, yy), textcoords="data", va="center", fontsize=10)
ax2.set_xticks(k)
ax2.set_xlim(0.7, 6.1)
ax2.set_ylim(0, 0.4)
ax2.set_xlabel("학습에 없던 두께인 층 수")
ax2.set_ylabel("MAE (nm)")
grid_y(ax2)
head(ax2, "동훈 브랜치 (test_final 2만 개)", "물리 피팅은 학습 격자와 무관하게 0.3 nm 안팎")
fig.tight_layout(w_pad=4)
save(fig, "fig6_unseen_layers.png", "두 그림은 세로축 단위(nm)는 같지만 범위가 100배 다르다.")

# ============ 그림 7. 신경망 본체 비교 ============
arch = [("1D ResNet", 26.0, 9.37), ("CNN + Transformer", 36.8, 11.57), ("MLP", 42.8, 13.53), ("FNO", 61.0, 17.12)]
fig, ax = plt.subplots(figsize=(9.5, 4.6))  # HANDOFF.md 6.7, mock_nn 51,712개, 신경망만, 10에폭
y = np.arange(len(arch))[::-1]
for yi, (n, o, m) in zip(y, arch):
    ax.barh(yi, o, height=0.6, color=BLUE if n == "1D ResNet" else GRAY, edgecolor=SURF, linewidth=2)
    ax.annotate(f"{o:.1f}%", (o, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=11)
    ax.annotate(f"MAE {m:.2f} nm", (o, yi), xytext=(52, 0), textcoords="offset points", va="center",
                fontsize=9.5, color=INK2)
ax.set_yticks(y, [a[0] for a in arch])
ax.set_xlim(0, 82)
ax.set_xlabel("한 층이라도 20 nm 넘게 틀린 비율 (%)")
ax.xaxis.grid(True)
ax.set_axisbelow(True)
head(ax, "신경망 본체 비교: 1D ResNet 선택", "같은 조건 10에폭, TMM 합성 데이터 50% (mock_nn 51,712개, 신경망만)")
save(fig, "fig7_architecture.png",
     "FNO는 곡선→곡선 문제에 강하지만 이 문제는 숫자 4개를 내는 회귀라 불리했다. 최종 ResNet은 30에폭으로 10.0%.")

# ============ 그림 8. mock vs test (과적합 확인) ============
fig, ax = plt.subplots(figsize=(10, 5))
xx = np.arange(3)
vm = [MOCK[k]["multistart"]["mae_avg"] for k in KEYS]
vt = [TF[k]["multistart"]["mae_avg"] for k in KEYS]
for off_, vals, c in ((-0.19, vm, GRAY), (0.19, vt, BLUE)):
    bars = ax.bar(xx + off_, vals, 0.36, color=c)
    for bb, vv in zip(bars, vals):
        ax.annotate(f"{vv:.3f}", (bb.get_x() + bb.get_width() / 2, vv), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10.5)
ax.set_xticks(xx, [NAMES[k] for k in KEYS])
ax.set_ylim(0, 0.4)
ax.set_ylabel("최종 MAE (nm)")
grid_y(ax)
ax.legend(handles=[Patch(color=GRAY, label="mock_refine 1만 개 (모델 고를 때 사용)"),
                   Patch(color=BLUE, label="test_final 2만 개 (마지막에 한 번)")], loc="upper left", ncol=2)
head(ax, "모의고사와 최종 시험 결과가 같다", "모델을 고르는 데 쓴 세트에 과적합되지 않았다 (차이 0.01 nm 이내)")
save(fig, "fig8_mock_vs_test.png")

# ============ 그림 9. 예시 곡선 (practice) ============
P = load_params()
par = params_tuple(P)
lam = P[:, 6]
grid = np.load(os.path.join(WORK, "grid.npy"), mmap_mode="r").reshape(-1, 226)
ev = np.load(os.path.join(WORK, "eval_sets.npz"))
prac = np.sort(ev["practice"])[:300]
Y = np.asarray(grid[prac], dtype=float)
D = (np.stack(np.unravel_index(prac, (30,) * 4), 1) + 1) * 10.0
from models import load_run  # noqa: E402
pred = load_run(os.path.join(WORK, "runs", "s3_resnet"))(Y.astype(np.float32))
d_fit = refine_lm(par, pred, Y)
i = int(np.argmin(np.abs(np.abs(d_fit - D).mean(1) - 0.18)))   # 피팅 오차가 test 중앙값(약 0.18 nm)에 가까운 샘플
fit = tmm(par, d_fit[i:i + 1])[0]
nnc = tmm(par, pred[i:i + 1])[0]
wrong_flat, w_true, w_found = 90996, [40, 120, 40, 70], [50.3, 104.9, 57.0, 47.0]  # HANDOFF.md 14.3 (practice)
yw = np.asarray(grid[wrong_flat], dtype=float)
ct = tmm(par, np.array([w_true], float))[0]
cf = tmm(par, np.array([w_found], float))[0]

fig = plt.figure(figsize=(14, 6.6))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.08, wspace=0.16)
a1, r1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
a2, r2 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
a1.plot(lam, Y[i], ".", color=MUTED, ms=4, label="측정 반사율 (노이즈 포함)")
a1.plot(lam, nnc, color=BLUE_L, lw=1.6, ls="--", label=f"신경망 답 {np.round(pred[i], 1).tolist()} nm")
a1.plot(lam, fit, color=BLUE, lw=2, label=f"TMM 피팅 {np.round(d_fit[i], 1).tolist()} nm")
head(a1, "① 신경망 답을 TMM 피팅으로 다듬기",
     f"정답 {D[i].astype(int).tolist()} nm · 신경망 최대 {np.abs(pred[i] - D[i]).max():.0f} nm 틀림 → "
     f"피팅 후 평균 {np.abs(d_fit[i] - D[i]).mean():.2f} nm", size=13)
r1.plot(lam, Y[i] - fit, ".", color=BLUE, ms=3)
r1.set_ylabel("잔차")
a2.plot(lam, yw, ".", color=MUTED, ms=4, label="측정 반사율")
a2.plot(lam, ct, color=BLUE, lw=2, label=f"정답 {w_true} nm")
a2.plot(lam, cf, color=ORANGE, lw=2, ls="--", label=f"찾은 답 {w_found} nm")
head(a2, "② 곡선은 같은데 두께가 다른 경우",
     "최대 23 nm 다르지만 광학 두께가 거의 같아 곡선 차이가 노이즈 수준", size=13)
r2.plot(lam, ct - cf, color=ORANGE, lw=1.5)
r2.set_ylabel("곡선 차이")
for a in (a1, a2):
    a.set_ylabel("반사율")
    a.tick_params(labelbottom=False)
    a.legend(loc="upper right", fontsize=9.5)
    grid_y(a)
for r in (r1, r2):
    r.axhspan(-SIGMA, SIGMA, color=GRID, zorder=0)
    r.axhline(0, color=MUTED, lw=1)
    r.annotate("±σ 노이즈", (lam.max(), SIGMA), xytext=(0, 2), textcoords="offset points", fontsize=8.5,
               color=INK2, ha="right", va="bottom")
    r.set_xlabel("파장 (nm)")
    r.set_ylim(-0.04, 0.04)
fig.suptitle("반사율 곡선과 TMM 피팅 예시", x=0.01, y=1.04, ha="left", fontsize=15, fontweight="bold")
save(fig, "fig9_example_curves.png", "practice 세트. 파장 축은 데이터에서 복원한 값(E = 1.550 + 0.0124·w eV).")
print("완료")
