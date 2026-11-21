# 1단계: train.csv를 grid.npy (30,30,30,30,226)로 변환. 인덱스 0=10 nm, 29=300 nm
import os, sys
import numpy as np, pandas as pd
from common import PROJECT, WORK
sys.stdout.reconfigure(encoding="utf-8")

tr = pd.read_csv(os.path.join(PROJECT, "train.csv"), dtype=np.float32)
L = tr.iloc[:, :4].values
X = tr.iloc[:, 4:].values
idx = ((L / 10 - 1).round().astype(int) * np.array([27000, 900, 30, 1])).sum(1)
order = np.argsort(idx)
assert (idx[order] == np.arange(810000)).all()
G = X[order].reshape(30, 30, 30, 30, 226)
np.save(os.path.join(WORK, "grid.npy"), G)
print("grid", G.shape)

# 노이즈 추정: 파장 축 4차 차분의 분산 = 70*sigma^2
d4 = np.diff(X[::50], n=4, axis=1)
print("noise sigma (4차 차분):", float(np.sqrt((d4 ** 2).mean() / 70)))
print("noise sigma (MAD):", float(1.4826 * np.median(np.abs(d4)) / np.sqrt(70)))
