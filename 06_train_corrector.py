# 4단계: 보정 신경망 학습 (TMM을 신경망 안에서 쓰는 구조)
#   python 06_train_corrector.py --base s3_resnet --name s4_corr
# 흐름: 기본 모델(고정) 답 d0 -> [TMM으로 곡선 R 그림 -> 보정 신경망이 (실제 곡선, R, 차이, 지금 두께)를 보고 고칠 양 냄] x iters
# 채점은 반복마다 '고친 두께 vs 정답 두께'로 한다. TMM 곡선은 입력으로만 쓰고 손실에 넣지 않는다
#   (곡선 점수를 손실에 넣은 FNO+TMM은 성능이 나빴다: HANDOFF.md 6.1).
# 반복 사이에는 기울기를 끊는다(detach). 기본 모델 답이 다양하게 틀린 경우를 겪도록 일부 샘플은 일부러 흔든다.
# 데이터: 04_train.py와 같이 실제 Train 반 + TMM 합성 반. best 기준: 고르기용 Valid 1만 개에서
#   '마지막 반복 후 곡선이 노이즈 수준(TOL_SEARCH 미만)이 아닌 비율' (= 정련이 필요한 비율).
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn
from common import WORK, SIGMA, TOL, TOL_SEARCH, load_params
from tmm_torch import TMMDecoder
from models import Corrector, build_model
sys.stdout.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="s3_resnet")
ap.add_argument("--name", required=True)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--iters", type=int, default=2, help="학습 때 보정 반복 횟수")
ap.add_argument("--synth-frac", type=float, default=0.5)
ap.add_argument("--perturb", type=float, default=15.0, help="흔들기 크기 상한 (nm). 샘플 절반에 N(0, U(0,perturb)) 더함")
ap.add_argument("--mse-w", type=float, default=0.1)
ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--n-valid", type=int, default=0, help="0이면 고르기용 1만 개 전부")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
dev = "cpu"

G = np.load(os.path.join(WORK, "grid.npy"), mmap_mode="r").reshape(810000, 226)
sp = np.load(os.path.join(WORK, "split.npz"))
sel = np.load(os.path.join(WORK, "eval_sets.npz"))["select"]
sel = sel[:a.n_valid] if a.n_valid else sel


def load(flat):
    flat = np.sort(flat)
    X = np.asarray(G[flat], dtype=np.float32)
    D = ((np.stack(np.unravel_index(flat, (30,) * 4), 1) + 1) * 10).astype(np.float32)
    return torch.tensor(X), torch.tensor(D)


Xtr, Dtr = load(sp["train"])
Xva, Dva = load(sel)
ck = torch.load(os.path.join(WORK, "runs", a.base, "best.pt"), weights_only=False, map_location="cpu")
base = build_model(ck["args"]); base.load_state_dict(ck["model"]); base.eval()
for p_ in base.parameters():
    p_.requires_grad_(False)
X_MU, X_SD, D_MU, D_SD = ck["stats"]
assert ck["args"].get("k", 1) == 1, "기본 모델은 후보 1개짜리여야 한다"
corr = Corrector()
dec = TMMDecoder(load_params(), dtype=torch.float32)
gen = TMMDecoder(load_params())
opt = torch.optim.AdamW(corr.parameters(), lr=a.lr, weight_decay=1e-4)
steps = (len(Xtr) + a.batch - 1) // a.batch
steps = min(steps, a.max_steps) if a.max_steps else steps
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps)
W = [0.5 ** (a.iters - 1 - t) for t in range(a.iters)]          # 뒤 반복일수록 비중 큼
print(f"base {a.base}, train {len(Xtr)}, valid {len(Xva)}, params {sum(p.numel() for p in corr.parameters()):,}, "
      f"steps/epoch {steps}, iters {a.iters}", flush=True)


def base_pred(x):
    return base((x - X_MU) / X_SD) * D_SD + D_MU


def step(x, d):
    R = dec(d.clamp(0.5, 400))
    return d + corr((x - X_MU) / X_SD, (R - X_MU) / X_SD, (x - R) / (10 * SIGMA), (d - D_MU) / D_SD) * D_SD


@torch.no_grad()
def evaluate(T, bs=2048):
    corr.eval()
    hist = [[] for _ in range(T + 1)]
    for s in range(0, len(Xva), bs):
        x = Xva[s:s + bs]
        d = base_pred(x); hist[0].append(d)
        for t in range(T):
            d = step(x, d); hist[t + 1].append(d)
    out = []
    for t in range(T + 1):
        P = torch.cat(hist[t]); e = (P - Dva).abs()
        r = ((dec(P.clamp(0.5, 400)) - Xva) ** 2).mean(1).sqrt()
        out.append({"iter": t, "mae_avg": e.mean().item(), "over20": (e.amax(1) > 20).float().mean().item(),
                    "over5": (e.amax(1) > 5).float().mean().item(),
                    "need_refine": (r >= TOL_SEARCH).float().mean().item(),
                    "conv_tol": (r < TOL).float().mean().item(),
                    "mae_conv": e[r < TOL_SEARCH].mean().item() if (r < TOL_SEARCH).any() else None})
    return out


out_dir = os.path.join(WORK, "runs", a.name); os.makedirs(out_dir, exist_ok=True)
log, best, t0 = [], 1e9, time.time()
ev0 = evaluate(a.iters)[0]
print(f"기본 모델만: MAE {ev0['mae_avg']:.3f}, >20nm {ev0['over20'] * 100:.1f}%, 정련 필요 {ev0['need_refine'] * 100:.1f}%", flush=True)
for ep in range(a.epochs):
    corr.train()
    perm = torch.randperm(len(Xtr))
    n_syn = int(round(a.batch * a.synth_frac)); n_real = a.batch - n_syn
    tot = 0.0
    for st in range(steps):
        b = perm[(st * n_real) % len(Xtr):][:n_real]
        x, d = Xtr[b], Dtr[b]
        with torch.no_grad():
            if n_syn:
                ds = torch.rand(n_syn, 4, dtype=torch.float64) * 280 + 10
                xs = gen(ds) + SIGMA * torch.randn(n_syn, 226, dtype=torch.float64)
                x, d = torch.cat([x, xs.float()]), torch.cat([d, ds.float()])
            dc = base_pred(x)
            m = (torch.rand(len(x), 1) < 0.5).float() * torch.rand(len(x), 1) * a.perturb
            dc = dc + m * torch.randn_like(dc)
        loss = 0.0
        for t in range(a.iters):
            dn = step(x, dc)
            e = dn - d
            loss = loss + W[t] * ((e.abs() / D_SD).mean() + a.mse_w * ((e / 20) ** 2).mean())
            dc = dn.detach()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(corr.parameters(), 1.0)
        opt.step(); sched.step()
        tot += loss.item()
    ev = evaluate(a.iters)
    rec = {"epoch": ep + 1, "loss": tot / steps, "valid_iters": ev, "valid": ev[-1], "time_s": round(time.time() - t0, 1)}
    log.append(rec)
    f = ev[-1]
    print(f"ep {ep + 1:3d} loss {rec['loss']:.4f} | 반복 {a.iters}번 후 MAE {f['mae_avg']:.3f} nm, >20nm {f['over20'] * 100:.1f}%, "
          f">5nm {f['over5'] * 100:.1f}%, 정련 필요 {f['need_refine'] * 100:.1f}% | 반복별 MAE "
          f"{[round(v['mae_avg'], 2) for v in ev]} | {rec['time_s']:.0f}s", flush=True)
    if f["need_refine"] < best:
        best = f["need_refine"]
        torch.save({"model": corr.state_dict(), "args": dict(vars(a), arch="corrector"),
                    "stats": [X_MU, X_SD, D_MU, D_SD]}, os.path.join(out_dir, "best.pt"))
    with open(os.path.join(out_dir, "log.json"), "w", encoding="utf-8") as fh:
        json.dump(log, fh, ensure_ascii=False, indent=1)
print(f"best 정련 필요 비율 {best:.4f}, total {time.time() - t0:.0f}s")
