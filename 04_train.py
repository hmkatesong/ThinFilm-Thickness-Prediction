# 4단계: FNO 단독 / FNO+TMM 학습
#   python 04_train.py --mode fno       --name fno
#   python 04_train.py --mode fno_tmm   --name fno_tmm
#   python 04_train.py --mode fno --name fno_synth --synth-frac 0.5 --mse-w 0.1 --select over20   (B+C)
# 손실: MAE(d_hat, d)/D_STD + beta * MSE(R_hat, R)/SIGMA^2   (fno 모드는 beta=0)
# Valid는 매 에폭 평가(라벨은 평가에만 사용). Test는 이 스크립트에서 쓰지 않는다.
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn
from common import WORK, SIGMA, load_params
from tmm_torch import TMMDecoder
from models import ARCHS, build_model
sys.stdout.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["fno", "fno_tmm"], required=True,
                help="fno: 두께 점수만 (모델 종류와 무관), fno_tmm: 곡선 점수 추가")
ap.add_argument("--arch", choices=ARCHS, default="fno", help="본체 모델 종류 (2단계 비교용)")
ap.add_argument("--name", required=True)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--width", type=int, default=64)
ap.add_argument("--modes", type=int, default=32)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--beta", type=float, default=0.02, help="곡선 점수 비중 (fno_tmm만)")
ap.add_argument("--warmup", type=int, default=10, help="beta를 0으로 두는 초기 에폭 수")
ap.add_argument("--n-train", type=int, default=0, help="0이면 Train 전체, 아니면 무작위 일부")
ap.add_argument("--n-valid", type=int, default=10000)
ap.add_argument("--max-steps", type=int, default=0, help="스모크 테스트용 에폭당 최대 스텝")
ap.add_argument("--seed", type=int, default=0)
# B. 물리 기반 데이터 늘리기: 배치의 일부를 TMM으로 만든 연속 두께 곡선 + 노이즈로 채운다.
#    굴절률은 Train만으로 구한 값이고, Valid·Test 정답은 쓰지 않는다.
ap.add_argument("--synth-frac", type=float, default=0.0, help="배치 중 합성 데이터 비율 (0이면 사용 안 함)")
ap.add_argument("--synth-lo", type=float, default=10.0)
ap.add_argument("--synth-hi", type=float, default=290.0, help="Train 두께 범위 안에서만 생성")
# C. 큰 오차 억제: MAE에 (오차/20nm)^2 항을 더한다. 정련은 출발점 오차가 20 nm 안이면 성공하므로.
ap.add_argument("--mse-w", type=float, default=0.0, help="(오차/mse-scale)^2 항의 비중")
ap.add_argument("--mse-scale", type=float, default=20.0)
ap.add_argument("--select", choices=["mae", "over20"], default="mae",
                help="best 모델 기준: 평균 오차 또는 '한 층이라도 20 nm 넘게 틀린 비율'")
# 3단계-a. 후보 여러 개 출력: 두께 후보 k세트를 내고, 학습 때는 정답에 가장 가까운 후보만 채점한다
#   (곡선이 비슷한 답이 둘이면 하나만 낼 때 둘의 중간으로 뭉개지는 문제). 나머지 후보도 굶지 않게 wta-eps만큼 나눠 준다.
#   예측 때는 후보마다 TMM 곡선을 그려 실제 곡선과 가장 잘 맞는 후보를 고른다 (정답 라벨 안 씀).
ap.add_argument("--k", type=int, default=1, help="두께 후보 수 (1이면 예전과 같음)")
ap.add_argument("--wta-eps", type=float, default=0.05)
# 3단계-b. 어려운 조합 더 연습: 합성 데이터 중 이 비율을 어려운 조합으로 만든다 (절반씩)
#   ① 사이 층이 얇은 조합: L2 또는 L3를 10~60 nm (같은 재료 두 층이 한 층처럼 보이는 경우)
#   ② 얇은 스택: 네 층 모두 10~130 nm
#   평가는 그대로 실제 Valid 분포로 한다.
ap.add_argument("--synth-hard", type=float, default=0.0)
a = ap.parse_args()
if a.k > 1 and a.mode != "fno":
    raise SystemExit("--k > 1은 --mode fno(두께 점수만)에서만 지원")
torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 데이터 ----------
G = np.load(os.path.join(WORK, "grid.npy"), mmap_mode="r").reshape(810000, 226)
sp = np.load(os.path.join(WORK, "split.npz"))
tr_idx = sp["train"] if not a.n_train else rng.choice(sp["train"], a.n_train, replace=False)
va_idx = rng.choice(sp["valid"], a.n_valid, replace=False)


def load(flat):
    flat = np.sort(flat)
    X = np.asarray(G[flat], dtype=np.float32)
    D = ((np.stack(np.unravel_index(flat, (30,) * 4), 1) + 1) * 10).astype(np.float32)
    return torch.tensor(X), torch.tensor(D)


Xtr, Dtr = load(tr_idx)
Xva, Dva = load(va_idx)
X_MU, X_SD = Xtr.mean(0), Xtr.std(0)          # 파장별 표준화 (Train 통계)
D_MU, D_SD = Dtr.mean(0), Dtr.std(0)          # 두께 정규화 (Train 통계)
print(f"device {dev}, train {len(Xtr)}, valid {len(Xva)}, mode {a.mode}, arch {a.arch}", flush=True)


# ---------- 모델 ----------
model = build_model(vars(a)).to(dev)
dec = TMMDecoder(load_params(), dtype=torch.float32).to(dev)
gen = TMMDecoder(load_params()).to(dev)        # 합성 데이터 생성용 (float64)
X_MU, X_SD, D_MU, D_SD = X_MU.to(dev), X_SD.to(dev), D_MU.to(dev), D_SD.to(dev)
n_par = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
steps_per_ep = (len(Xtr) + a.batch - 1) // a.batch
if a.max_steps:
    steps_per_ep = min(steps_per_ep, a.max_steps)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps_per_ep)


def predict_all(X):
    """(B, k, 4) nm 두께 후보."""
    return model((X - X_MU) / X_SD).view(len(X), a.k, 4) * D_SD + D_MU


def predict(X):
    return predict_all(X)[:, 0]   # k=1일 때 예전과 같음 (곡선 점수 손실용)


@torch.no_grad()
def choose(C, x):
    """후보 중 TMM 곡선이 실제 곡선 x와 가장 잘 맞는 것 (B, 4)과 그 곡선 MSE."""
    B = len(C)
    R = dec(C.reshape(-1, 4).clamp(0.5, 400)).view(B, a.k, -1)
    mse = ((R - x[:, None]) ** 2).mean(2)
    j = mse.argmin(1)
    return C[torch.arange(B), j], mse[torch.arange(B), j]


@torch.no_grad()
def evaluate(X, D, bs=2048):
    model.eval()
    P, Rm, Orc = [], [], []
    for s in range(0, len(X), bs):
        x, d = X[s:s + bs].to(dev), D[s:s + bs].to(dev)
        C = predict_all(x)
        p, m = choose(C, x)
        P.append(p.cpu()); Rm.append(m.cpu())
        Orc.append((C - d[:, None]).abs().amax(2).amin(1).cpu())
    P = torch.cat(P)
    err = (P - D).abs()
    return {"mae": err.mean(0).tolist(), "mae_avg": err.mean().item(),
            "over20": (err.amax(1) > 20).float().mean().item(),
            "oracle_over20": (torch.cat(Orc) > 20).float().mean().item(),
            "rmse": ((P - D) ** 2).mean(0).sqrt().tolist(),
            "curve_rmse": torch.cat(Rm).mean().sqrt().item()}


def sample_synth(n):
    """합성 두께 (n, 4). synth-hard 비율만큼 어려운 조합."""
    lo, hi = a.synth_lo, a.synth_hi
    ds = torch.rand(n, 4, device=dev, dtype=torch.float64) * (hi - lo) + lo
    if a.synth_hard > 0:
        u = torch.rand(n, device=dev)
        mid = u < a.synth_hard / 2                                 # ① 사이 층 얇게
        thin = (u >= a.synth_hard / 2) & (u < a.synth_hard)        # ② 얇은 스택
        which = torch.randint(1, 3, (n,), device=dev)              # L2 또는 L3
        tval = torch.rand(n, device=dev, dtype=torch.float64) * 50 + 10
        rows = torch.nonzero(mid).squeeze(1)
        ds[rows, which[rows]] = tval[rows]
        ds[thin] = torch.rand(int(thin.sum()), 4, device=dev, dtype=torch.float64) * 120 + 10
    return ds


out_dir = os.path.join(WORK, "runs", a.name)
os.makedirs(out_dir, exist_ok=True)
log = []
best = 1e9
print(f"params {n_par:,}, steps/epoch {steps_per_ep}", flush=True)
t0 = time.time()
for ep in range(a.epochs):
    model.train()
    # 처음에는 두께 점수만으로 대략 맞힌 뒤, 곡선 점수를 5에폭에 걸쳐 서서히 올린다.
    # (추측이 멀리 틀린 상태에서는 곡선 점수가 출렁여서 학습을 방해함: 스모크 테스트에서 확인)
    ramp = min(1.0, max(0.0, (ep - a.warmup + 1) / 5))
    beta = 0.0 if a.mode == "fno" else a.beta * ramp
    perm = torch.randperm(len(Xtr))
    tot = tot_d = tot_r = 0.0
    n_syn = int(round(a.batch * a.synth_frac))
    n_real = a.batch - n_syn
    for st in range(steps_per_ep):
        b = perm[(st * n_real) % len(Xtr):][:n_real]
        x, d = Xtr[b].to(dev), Dtr[b].to(dev)
        if n_syn:
            with torch.no_grad():
                ds = sample_synth(n_syn)
                xs = gen(ds) + SIGMA * torch.randn(n_syn, 226, device=dev, dtype=torch.float64)
            x, d = torch.cat([x, xs.float()]), torch.cat([d, ds.float()])
        C = predict_all(x)                                  # (B, k, 4)
        p = C[:, 0]
        e = C - d[:, None]
        lk = (e.abs() / D_SD).mean(2)                       # 후보별 점수 (B, k)
        if a.mse_w > 0:
            lk = lk + a.mse_w * ((e / a.mse_scale) ** 2).mean(2)
        if a.k == 1:
            loss_d = lk[:, 0].mean()
        else:                                               # 가장 가까운 후보만 채점 + 조금은 전체에
            loss_d = ((1 - a.wta_eps) * lk.min(1).values + a.wta_eps * lk.mean(1)).mean()
        loss = loss_d
        if beta > 0:
            loss_r = ((dec(p) - x) ** 2).mean() / SIGMA ** 2
            loss = loss + beta * loss_r
            tot_r += loss_r.item()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        tot += loss.item(); tot_d += loss_d.item()
    ev = evaluate(Xva, Dva)
    rec = {"epoch": ep + 1, "beta": beta, "loss": tot / steps_per_ep, "loss_d": tot_d / steps_per_ep,
           "loss_r": tot_r / steps_per_ep, "valid": ev, "time_s": round(time.time() - t0, 1)}
    log.append(rec)
    print(f"ep {ep + 1:3d} beta {beta:.3f} loss {rec['loss']:.4f} | valid MAE "
          f"{[round(v, 2) for v in ev['mae']]} avg {ev['mae_avg']:.3f} nm, >20nm {ev['over20'] * 100:.1f}% "
          f"(후보 중 최선 {ev['oracle_over20'] * 100:.1f}%), "
          f"curve RMSE {ev['curve_rmse']:.4f} | {rec['time_s']:.0f}s", flush=True)
    score = ev["mae_avg"] if a.select == "mae" else ev["over20"]
    if score < best:
        best = score
        torch.save({"model": model.state_dict(), "args": vars(a),
                    "stats": [t.cpu() for t in (X_MU, X_SD, D_MU, D_SD)]}, os.path.join(out_dir, "best.pt"))
    with open(os.path.join(out_dir, "log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
print(f"best valid {a.select} {best:.4f}, total {time.time() - t0:.0f}s")
