"""피처 만들기. 모두 행(샘플) 하나 안에서만 계산한다. PCA만 train으로 fit한다.

그룹 이름과 뜻 (발표용 한 줄 설명):
  raw   원시 반사율 226개
  sg    Savitzky-Golay 1·2차 미분: 잡음을 줄이며 곡선 기울기와 굽은 정도를 꺼냄 (분광 분석 표준 전처리)
  fft   푸리에 변환 저주파 30개 성분: '곡선이 몇 번 출렁이나'를 숫자로 꺼냄 (막이 두꺼울수록 출렁임이 촘촘)
  peak  봉우리·골짜기 개수, 위치, 높이, 간격 (scipy find_peaks)
  band  226점을 8구간으로 나눠 구간별 평균·표준편차 + 전체 요약값
  pca   주성분 20개 (곡선 모양을 20개 숫자로 압축)
"""
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from sklearn.decomposition import PCA

ALL_GROUPS = ["raw", "sg", "fft", "peak", "band", "pca"]
SG_WINDOW, SG_POLY = 11, 3
N_FFT = 30
N_BAND = 8
N_PCA = 20
PEAK_PROM = 0.03  # 잡음 표준편차 약 0.009의 3배 정도: 잡음 출렁임은 봉우리로 안 셈
N_PEAK_KEEP = 4


def smooth(X):
    return savgol_filter(X, SG_WINDOW, SG_POLY, axis=1).astype(np.float32)


def f_sg(X):
    d1 = savgol_filter(X, SG_WINDOW, SG_POLY, deriv=1, axis=1)
    d2 = savgol_filter(X, SG_WINDOW, SG_POLY, deriv=2, axis=1)
    return np.hstack([d1, d2]).astype(np.float32)


def f_fft(S):
    """S: 평활화된 곡선. 평균을 빼고 Hann 창을 씌운 뒤 FFT (끝부분 잘림 효과를 줄이는 표준 처리)."""
    Z = (S - S.mean(axis=1, keepdims=True)) * np.hanning(S.shape[1])
    F = np.fft.rfft(Z, axis=1)[:, 1 : N_FFT + 1]
    return np.hstack([np.abs(F), F.real, F.imag]).astype(np.float32)


def _row_peaks(s):
    out = np.full(3 + 4 * N_PEAK_KEEP, np.nan, dtype=np.float32)
    p, _ = find_peaks(s, prominence=PEAK_PROM)
    v, _ = find_peaks(-s, prominence=PEAK_PROM)
    out[0], out[1] = len(p), len(v)
    ext = np.sort(np.concatenate([p, v]))
    if len(ext) >= 2:
        out[2] = np.diff(ext).mean()  # 출렁임 반주기 평균 간격
    k = 3
    for idx in (p, v):
        m = min(len(idx), N_PEAK_KEEP)
        out[k : k + m] = idx[:m]
        out[k + N_PEAK_KEEP : k + N_PEAK_KEEP + m] = s[idx[:m]]
        k += 2 * N_PEAK_KEEP
    return out


def f_peak(S):
    return np.vstack([_row_peaks(s) for s in S])


def f_band(S):
    parts = np.array_split(S, N_BAND, axis=1)
    mean = np.stack([b.mean(axis=1) for b in parts], axis=1)
    std = np.stack([b.std(axis=1) for b in parts], axis=1)
    glob = np.stack(
        [S.mean(1), S.std(1), S.min(1), S.max(1), S.argmin(1), S.argmax(1)], axis=1
    )
    return np.hstack([mean, std, glob]).astype(np.float32)


class FeatureBuilder:
    """fit(X_train) 후 transform(X). PCA 외에는 학습할 것이 없다."""

    def __init__(self, groups):
        bad = set(groups) - set(ALL_GROUPS)
        if bad:
            raise ValueError(f"모르는 피처 그룹: {bad}")
        self.groups = list(groups)
        self.pca = None

    def fit(self, X):
        if "pca" in self.groups:
            self.pca = PCA(N_PCA, random_state=0).fit(smooth(X))
        return self

    def transform(self, X):
        S = smooth(X)
        out = []
        for g in self.groups:
            if g == "raw":
                out.append(X.astype(np.float32))
            elif g == "sg":
                out.append(f_sg(X))
            elif g == "fft":
                out.append(f_fft(S))
            elif g == "peak":
                out.append(f_peak(S))
            elif g == "band":
                out.append(f_band(S))
            elif g == "pca":
                out.append(self.pca.transform(S).astype(np.float32))
        return np.hstack(out)
