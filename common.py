# 공통 설정과 numpy TMM 함수 (인수인계 문서 부록 A.1 기반)
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)                      # train.csv 위치
WORK = os.path.join(HERE, "_work")                   # 중간 산출물
os.makedirs(WORK, exist_ok=True)
SIGMA = 0.0087        # 데이터 노이즈 추정값
# 곡선 RMSE 기준 두 가지 (diag_noise.py --quick: Train 50,625개 정답 곡선 RMSE 99%=0.0104, 99.9%=0.0108, 최대 0.0112)
# TOL: 최종 판정 '수렴(믿을 만함)'. 정답이면 거의 전부 이 안에 든다. 예전 0.0105는 맞힌 샘플도 약 0.5%를 실패로 셌다.
# TOL_SEARCH: 이보다 크면 다시 찾기를 계속한다. 0.0115짜리 틀린 답(정답은 0.0088)도 있었으므로
#             찾기는 낮은 기준으로 끝까지 하고, 판정만 TOL로 한다. (다시 찾기는 곡선이 더 잘 맞는 답만 받아들임)
TOL = 0.0120
TOL_SEARCH = 0.0105
TOL_OLD = 0.0105      # 예전 판정 기준 (비교용)
HC = 1239.84          # eV*nm
ORDER = (3, 2, 1, 0)  # 기판 쪽(layer_4)에서 공기 쪽(layer_1)으로


def load_params(path=None):
    """광학 상수 CSV를 읽어 (226,7) 배열 [nH,kH,nL,kL,ns,ks,lam] 반환."""
    path = path or os.path.join(WORK, "optical_params.csv")
    return np.loadtxt(path, delimiter=",", skiprows=1)[:, 1:8]


def params_tuple(P):
    return tuple(P[:, k] for k in range(7))


def tmm(par, d):
    """par: (nH,kH,nL,kL,ns,ks,lam) 각 스칼라 또는 (W,) 배열, d: (N,4) nm. 반환 R: (N,) 또는 (N,W)."""
    nH, kH, nL, kL, ns, ks, lam = par
    NH, NL, Ns = nH - 1j * kH, nL - 1j * kL, ns - 1j * ks
    scalar = np.ndim(lam) == 0
    B = np.ones((d.shape[0],) if scalar else (d.shape[0], len(lam)), dtype=complex)
    C = B * Ns
    for j in ORDER:
        Nj = NH if j in (0, 2) else NL
        dj = d[:, j] if scalar else d[:, j:j + 1]
        dl = 2 * np.pi * Nj * dj / lam
        c, s = np.cos(dl), np.sin(dl)
        B, C = c * B + 1j * s / Nj * C, 1j * s * Nj * B + c * C
    r = (B - C) / (B + C)
    return r.real ** 2 + r.imag ** 2


def jacobian(par, d, h=0.5):
    """두께에 대한 중앙차분 야코비안 (N,W,4)."""
    J = []
    for j in range(4):
        e = np.zeros(4)
        e[j] = h
        J.append((tmm(par, d + e) - tmm(par, d - e)) / (2 * h))
    return np.stack(J, -1)


def refine(par, d0, y, iters=8, damp=1e-6):
    """샘플별 연속 두께 Gauss-Newton 정련. 격자 반올림 없음."""
    d = d0.astype(float).copy()
    for _ in range(iters):
        J = jacobian(par, d)
        r = y - tmm(par, d)
        A = np.einsum("nwi,nwj->nij", J, J) + damp * np.eye(4)
        g = np.einsum("nwi,nw->ni", J, r)
        d = np.clip(d + np.linalg.solve(A, g[..., None])[..., 0], 0.5, 400)
    return d


def curve_rmse(par, d, y):
    """두께 d로 다시 계산한 곡선과 실제 곡선 y의 샘플별 RMSE."""
    return np.sqrt(((tmm(par, d) - y) ** 2).mean(1))


def refine_lm(par, d0, y, iters=20, lo=5.0, hi=300.0):
    """개선 정련 (Levenberg-Marquardt). 곡선이 더 잘 맞을 때만 두께를 고치고, 아니면 보폭을 줄인다.
    기존 refine()은 나빠지는 이동도 받아들여 일부 샘플이 수십 nm 밖으로 튀었다(진단 결과)."""
    d = np.clip(d0, lo, hi).astype(float).copy()
    r = y - tmm(par, d)
    cost = (r ** 2).sum(1)
    lam = np.full(len(d), 1e-3)
    eye = np.eye(4)
    for _ in range(iters):
        J = jacobian(par, d)
        A = np.einsum("nwi,nwj->nij", J, J)
        g = np.einsum("nwi,nw->ni", J, r)
        Ad = A + lam[:, None, None] * (np.einsum("nii->ni", A)[:, :, None] * eye + 1e-9 * eye)
        dn = np.clip(d + np.linalg.solve(Ad, g[..., None])[..., 0], lo, hi)
        rn = y - tmm(par, dn)
        cn = (rn ** 2).sum(1)
        acc = cn < cost
        d[acc], r[acc], cost[acc] = dn[acc], rn[acc], cn[acc]
        lam = np.where(acc, lam * 0.3, lam * 10)
    return d


def refine_best(par, starts, y, iters=20):
    """출발점 여러 세트(list of (N,4))를 각각 LM 정련하고 샘플마다 곡선이 가장 잘 맞는 결과를 고른다."""
    d = refine_lm(par, starts[0], y, iters)
    r = curve_rmse(par, d, y)
    for s in starts[1:]:
        dn = refine_lm(par, s, y, iters)
        rn = curve_rmse(par, dn, y)
        u = rn < r
        d[u], r[u] = dn[u], rn[u]
    return d, r


def refine_multistart(par, d0, y, n_starts=12, spread=20.0, tol=TOL_SEARCH, iters=20, seed=3, pair_split=True,
                      swap=True, init=None):
    """개선 정련 후 곡선이 노이즈 수준(tol)까지 안 맞은 샘플만 다시 정련해 곡선이 가장 잘 맞는 답을 고른다.
    1) 출발점 주변 무작위 n_starts개
    2) pair_split: 같은 재료 쌍(L1+L3 SiN, L2+L4 SiO2)의 두께 합은 유지하고 나누는 비율만 바꾼 7x7 출발점.
       실패 샘플 분석 결과, 정련은 쌍의 '합'은 맞히지만(광학 두께 차 0.9%) 쌍 안에서 나누는 비율을 틀렸다.
       사이 층이 얇으면 같은 재료 두 층이 거의 한 층처럼 보이기 때문이다.
    정답 라벨은 쓰지 않는다.
    init=(d, r): 첫 LM 정련을 이미 했으면 그 결과를 넘겨 다시 하지 않는다 (05_eval.py가 사용)."""
    if init is not None:
        d, r = init[0].copy(), init[1].copy()
    else:
        d = refine_lm(par, d0, y, iters)
        r = curve_rmse(par, d, y)

    def try_starts(idx, starts):
        dn = refine_lm(par, starts, y[idx], iters)
        rn = curve_rmse(par, dn, y[idx])
        u = rn < r[idx]
        d[idx[u]], r[idx[u]] = dn[u], rn[u]

    fr = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)
    bad = np.flatnonzero(r >= tol)
    for o in np.random.default_rng(seed).normal(0, spread, (n_starts, 4)):
        if not len(bad):
            break
        try_starts(bad, d0[bad] + o)
    if pair_split:
        for src in (d.copy(), d0):
            bad = np.flatnonzero(r >= tol)
            if not len(bad):
                break
            s13 = np.clip(src[bad, 0] + src[bad, 2], 20, 580)
            s24 = np.clip(src[bad, 1] + src[bad, 3], 20, 580)
            for f1 in fr:
                for f2 in fr:
                    try_starts(bad, np.stack([f1 * s13, f2 * s24, (1 - f1) * s13, (1 - f2) * s24], 1))
    if swap:
        # 3) 재료 맞바꿈: 총 광학 두께 nH*(L1+L3) + nL*(L2+L4)는 유지하고 SiN:SiO2 비율을 바꾼 출발점.
        #    연습용 세트의 진짜 실패 3개는 SiN을 줄이고 SiO2를 늘려(또는 반대) 광학 두께를 유지한 가짜 답이었다.
        nH, nL = float(np.median(par[0])), float(np.median(par[2]))
        for src in (d.copy(), d0):
            bad = np.flatnonzero(r >= tol)
            if not len(bad):
                break
            opt = nH * (src[bad, 0] + src[bad, 2]) + nL * (src[bad, 1] + src[bad, 3])
            for g in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                s13 = np.clip(g * opt / nH, 20, 580)
                s24 = np.clip((1 - g) * opt / nL, 20, 580)
                for f1 in fr:          # 쌍 안 배분도 함께 바꿈 (실패 샘플은 290:10처럼 치우친 배분이 많았다)
                    for f2 in fr:
                        try_starts(bad, np.stack([f1 * s13, f2 * s24, (1 - f1) * s13, (1 - f2) * s24], 1))
    return d, r
