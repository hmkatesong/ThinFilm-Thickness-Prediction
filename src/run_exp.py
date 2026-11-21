"""실험 1개 실행 -> experiment_log.csv에 한 줄 추가.

예: python src/run_exp.py --exp E05 --model lgbm --feats raw,sg --change "+SG 미분"
시드를 여러 개 주면 같은 설정을 시드별로 돌려 한 줄로 남긴다.
  MAE평균, MAE표준편차 = 시드별 MAE의 평균과 표준편차(ddof=0). 분할은 seed 0으로 고정, 모델 시드만 바뀐다.
  시드평균예측MAE = 시드별 예측을 먼저 평균한 뒤 잰 MAE (시드 앙상블). 시드 1개면 MAE평균과 같다.
--eval test 는 최종 확인용이다. 설정을 고를 때는 valid만 쓴다.
이미 로그에 있는 --exp 이름은 거부한다 (재현할 때는 새 이름, 예: R14).
"""
import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data import load_train, ROOT  # noqa: E402
from features import FeatureBuilder  # noqa: E402
import models  # noqa: E402
from models import MODELS  # noqa: E402

DEFAULT_LR = models.LGBM_LR
from split import random_split, skip_split  # noqa: E402

LOG = ROOT / "experiment_log.csv"
PRED_DIR = ROOT / "results" / "preds"
COLS = ["실험", "날짜", "분할", "모델", "피처", "시드", "평가셋", "피처수",
        "MAE평균", "MAE표준편차", "MAE_L1", "MAE_L2", "MAE_L3", "MAE_L4",
        "시간초", "변경점", "메모", "시드평균예측MAE"]


def get_split(name, y, seed=0, n_train=None):
    if name == "skip":
        tr, va, te = skip_split(y, seed=seed)
        return tr, va, te
    kw = {"n_train": n_train} if n_train else {}
    tr, va = random_split(len(y), seed=seed, **kw)
    return tr, va, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--feats", default="raw")
    ap.add_argument("--split", default="skip", choices=["skip", "random"])
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--eval", default="valid", choices=["valid", "test"])
    ap.add_argument("--n_train", type=int, default=None, help="random 분할 학습 크기 (기본 50,625)")
    ap.add_argument("--lr", type=float, default=models.LGBM_LR, help="LightGBM 학습률 (기본 0.1)")
    ap.add_argument("--change", default="")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if LOG.exists():
        with open(LOG, encoding="utf-8-sig") as f:
            if any(r["실험"] == a.exp for r in csv.DictReader(f)):
                raise SystemExit(f"{a.exp}는 이미 로그에 있다. 새 이름을 써라 (예: R{a.exp[1:]})")
    models.LGBM_LR = a.lr
    if a.n_train:
        a.note = f"n_train {a.n_train} {a.note}".strip()
    if a.model == "lgbm" and a.lr != DEFAULT_LR:
        a.note = f"lr {a.lr} {a.note}".strip()

    t0 = time.time()
    X, y = load_train()
    tr, va, te = get_split(a.split, y, n_train=a.n_train)
    if a.eval == "test" and te is None:
        raise SystemExit("random 분할에는 test가 없다")
    ev = va if a.eval == "valid" else te

    groups = a.feats.split(",")
    fb = FeatureBuilder(groups).fit(X[tr])
    Ftr, Fva, Fev = fb.transform(X[tr]), fb.transform(X[va]), fb.transform(X[ev])

    seeds = [int(s) for s in a.seeds.split(",")]
    fit = MODELS[a.model]
    scores, per_layer, preds, infos = [], [], [], []
    for s in seeds:
        (p,), info = fit(Ftr, y[tr], Fva, y[va], [Fev], s)
        err = np.abs(y[ev] - p)
        scores.append(err.mean())
        per_layer.append(err.mean(axis=0))
        preds.append(p)
        infos.append(info)
        print(f"seed {s}: MAE {err.mean():.4f}  층별 {np.round(err.mean(0), 4)}  {info}", flush=True)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    ens = np.mean(preds, axis=0).astype(np.float32)
    np.save(PRED_DIR / f"{a.exp}_{a.eval}.npy", ens)
    np.save(PRED_DIR / f"{a.exp}_{a.eval}_idx.npy", ev)

    pl = np.mean(per_layer, axis=0)
    row = {
        "실험": a.exp, "날짜": dt.date.today().isoformat(), "분할": a.split,
        "모델": a.model, "피처": a.feats, "시드": a.seeds, "평가셋": a.eval,
        "피처수": Ftr.shape[1], "MAE평균": f"{np.mean(scores):.4f}",
        "MAE표준편차": f"{np.std(scores):.4f}" if len(seeds) > 1 else "",
        **{f"MAE_L{i+1}": f"{pl[i]:.4f}" for i in range(4)},
        "시간초": int(time.time() - t0), "변경점": a.change,
        "메모": (a.note + " " + json.dumps(infos, ensure_ascii=False)).strip(),
        "시드평균예측MAE": f"{np.abs(y[ev] - ens).mean():.4f}",
    }
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[{a.exp}] {a.model} {a.feats} {a.split}/{a.eval}: MAE {row['MAE평균']} "
          f"{row['MAE표준편차']} 시드평균예측 {row['시드평균예측MAE']} ({row['시간초']}s, 피처 {row['피처수']})")


if __name__ == "__main__":
    main()
