"""데이터 읽기. CSV(1.9 GB)는 처음 한 번만 읽고 cache/에 npy로 저장한다.

원본 CSV 위치 (먼저 찾는 순서):
  1. 환경변수 THINFILM_DATA 가 가리키는 폴더
  2. 레포 안 data/ (DACON 235554에서 train.csv, test.csv를 받아 여기에 둔다)
  3. 윤서 PC의 Wiki 원본 폴더 (이 레포가 Wiki/projects/... 아래에 있을 때만)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
TARGETS = ["layer_1", "layer_2", "layer_3", "layer_4"]
N_WAVE = 226


def raw_dir():
    """train.csv가 있는 폴더. 없으면 찾아본 곳을 알려 주고 멈춘다."""
    env = os.environ.get("THINFILM_DATA")
    if env:  # 환경변수를 줬으면 거기만 본다 (경로 오타를 다른 데이터로 덮지 않게)
        if not (Path(env) / "train.csv").exists():
            raise FileNotFoundError(f"THINFILM_DATA={env} 에 train.csv가 없다")
        return Path(env)
    cands = [ROOT / "data"]
    if len(ROOT.parents) > 3:
        cands.append(ROOT.parents[3] / "raw" / "competitions" / "dacon-235554-박막두께")
    for c in cands:
        if (c / "train.csv").exists():
            return c
    raise FileNotFoundError(
        "train.csv를 못 찾았다. DACON 235554 데이터를 data/에 두거나 환경변수 THINFILM_DATA로 "
        "폴더를 알려 줘. 찾아본 곳: " + ", ".join(str(c) for c in cands))


def _read_csv(path):
    head = pd.read_csv(path, nrows=0).columns
    dtypes = {c: np.float32 for c in head if c != "id"}
    return pd.read_csv(path, dtype=dtypes)


def load_train():
    """X: (810000, 226) float32 반사율, y: (810000, 4) float32 두께(nm)."""
    fx, fy = CACHE / "train_X.npy", CACHE / "train_y.npy"
    if not fx.exists():
        CACHE.mkdir(exist_ok=True)
        df = _read_csv(raw_dir() / "train.csv")
        np.save(fy, df[TARGETS].to_numpy(np.float32))
        np.save(fx, df[[str(i) for i in range(N_WAVE)]].to_numpy(np.float32))
    return np.load(fx), np.load(fy)


def load_test():
    """X: (10000, 226) float32, ids: (10000,) int."""
    fx, fi = CACHE / "test_X.npy", CACHE / "test_id.npy"
    if not fx.exists():
        CACHE.mkdir(exist_ok=True)
        df = _read_csv(raw_dir() / "test.csv")
        np.save(fi, df["id"].to_numpy())
        np.save(fx, df[[str(i) for i in range(N_WAVE)]].to_numpy(np.float32))
    return np.load(fx), np.load(fi)
