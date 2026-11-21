# 1D FNO 모델 (04_train.py와 05_eval.py가 같이 쓴다)
import os
import numpy as np
import torch, torch.nn as nn


class SpectralConv1d(nn.Module):
    """푸리에 변환 -> 낮은 모드만 남겨 모드별 복소 가중치 곱 -> 역변환."""

    def __init__(self, c, modes):
        super().__init__()
        self.modes = modes
        self.w = nn.Parameter(torch.randn(c, c, modes, dtype=torch.cfloat) / (c * c))

    def forward(self, x):                        # x: (B, C, N)
        xf = torch.fft.rfft(x)
        out = torch.zeros_like(xf)
        m = min(self.modes, xf.shape[-1])
        out[..., :m] = torch.einsum("bim,iom->bom", xf[..., :m], self.w[..., :m])
        return torch.fft.irfft(out, n=x.shape[-1])


class FNO1d(nn.Module):
    def __init__(self, width, modes, layers, pad=32):
        super().__init__()
        self.pad = pad
        self.lift = nn.Conv1d(2, width, 1)        # 입력 채널: 반사율, 에너지 좌표
        self.spec = nn.ModuleList(SpectralConv1d(width, modes) for _ in range(layers))
        self.pw = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(2 * width, 128), nn.GELU(), nn.Linear(128, 4))

    def forward(self, x):                        # x: (B, 226) 표준화된 반사율
        B, N = x.shape
        grid = torch.linspace(-1, 1, N, device=x.device).expand(B, N)
        h = self.lift(torch.stack([x, grid], 1))
        h = nn.functional.pad(h, (0, self.pad))   # 비주기 신호라 오른쪽을 덧댐
        for k, (sc, pw) in enumerate(zip(self.spec, self.pw)):
            h = sc(h) + pw(h)
            if k < len(self.spec) - 1:
                h = nn.functional.gelu(h)
        h = h[..., :N]
        z = torch.cat([h.mean(-1), h.amax(-1)], 1)  # 전역 요약
        return self.head(z)                        # 정규화된 두께


# ---------------- 2단계 후보 모델 ----------------
# 공통 규칙: 입력 (B, 226) 표준화된 반사율 -> 출력 (B, 4) 정규화된 두께.
# 마지막 층에서 평균/최댓값으로 뭉개지 않고, 위치별 특징을 펼쳐(flatten) 넣는다.
# (FNO는 평균·최댓값만 써서 봉우리 위치 정보가 사라졌다.)

class MLP(nn.Module):
    """반사율 226개를 통째로 보는 기본 신경망."""

    def __init__(self, n=226, hidden=512, depth=4, drop=0.0):
        super().__init__()
        layers, c = [], n
        for _ in range(depth):
            layers += [nn.Linear(c, hidden), nn.LayerNorm(hidden), nn.GELU()]
            if drop:
                layers.append(nn.Dropout(drop))
            c = hidden
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, 4)

    def forward(self, x):
        return self.out(self.body(x))


class ResBlock1d(nn.Module):
    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Identity() if cin == cout and stride == 1 else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)))

    def forward(self, x):
        h = nn.functional.gelu(self.b1(self.c1(x)))
        return nn.functional.gelu(self.b2(self.c2(h)) + self.skip(x))


def _with_grid(x):
    B, N = x.shape
    grid = torch.linspace(-1, 1, N, device=x.device).expand(B, N)
    return torch.stack([x, grid], 1)               # (B, 2, N): 반사율, 에너지 좌표


class ResNet1d(nn.Module):
    """작은 창으로 곡선을 훑는 합성곱을 깊게 쌓은 모델. 길이 226 -> 113 -> 57 -> 29 -> 15.
    가장 두꺼운 스택도 기본 출렁임 한 주기가 약 23점이고, 첫 합성곱(창 7)이 줄이기 전에 세부 모양을 읽는다."""

    def __init__(self, n=226, chans=(64, 96, 128, 128), blocks=2, head_c=32, out=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(2, chans[0], 7, 2, 3, bias=False), nn.BatchNorm1d(chans[0]), nn.GELU())
        n = (n + 1) // 2
        layers, c = [], chans[0]
        for i, co in enumerate(chans):
            for b in range(blocks):
                layers.append(ResBlock1d(c, co, 5 if i < 2 else 3, 2 if (b == 0 and i > 0) else 1))
                c = co
        self.body = nn.Sequential(*layers)
        self.reduce = nn.Conv1d(c, head_c, 1)
        L = n
        for _ in range(len(chans) - 1):
            L = (L + 1) // 2
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(head_c * L, 256), nn.GELU(), nn.Linear(256, out))

    def forward(self, x):
        h = self.body(self.stem(_with_grid(x)))
        return self.head(nn.functional.gelu(self.reduce(h)))


class CNNTransformer(nn.Module):
    """앞쪽 합성곱이 국소 모양(봉우리)을 읽어 29개 조각으로 요약하고,
    뒤쪽 Transformer가 조각끼리의 관계(봉우리 간격)를 본다. 조각 하나는 약 8점(0.1 eV)."""

    def __init__(self, n=226, d=128, layers=4, heads=4, ff=256, head_c=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, 5, 1, 2), nn.GELU(),
            nn.Conv1d(64, 96, 5, 2, 2), nn.GELU(),
            nn.Conv1d(96, d, 5, 2, 2), nn.GELU(),
            nn.Conv1d(d, d, 3, 2, 1), nn.GELU())
        L = (((n + 1) // 2 + 1) // 2 + 1) // 2      # 29
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        enc = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, activation="gelu",
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.reduce = nn.Linear(d, head_c)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(head_c * L, 256), nn.GELU(), nn.Linear(256, 4))

    def forward(self, x):
        t = self.stem(_with_grid(x)).transpose(1, 2) + self.pos   # (B, 57, d)
        t = self.norm(self.enc(t))
        return self.head(nn.functional.gelu(self.reduce(t)))


class Corrector(nn.Module):
    """4단계 보정 신경망. 지금 두께 d로 TMM 곡선 R을 그린 뒤, 실제 곡선 x와 비교해서 두께 고칠 양을 낸다.
    입력 채널 4개: 실제 곡선, 그린 곡선, 둘의 차이(노이즈 10배 단위), 에너지 좌표. 지금 두께(정규화)는 출력층 앞에 붙인다.
    정련(LM)이 층마다 두께를 바꿔 보며 방향을 찾는 일을, 곡선 차이 모양을 보고 바로 알아내도록 배운다.
    마지막 층을 0으로 시작해서 처음에는 '아무것도 안 고침'에서 출발한다."""

    def __init__(self, n=226, chans=(64, 96, 128, 128), blocks=2, head_c=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(4, chans[0], 7, 2, 3, bias=False), nn.BatchNorm1d(chans[0]), nn.GELU())
        L, layers, c = (n + 1) // 2, [], chans[0]
        for i, co in enumerate(chans):
            for b in range(blocks):
                layers.append(ResBlock1d(c, co, 5 if i < 2 else 3, 2 if (b == 0 and i > 0) else 1))
                c = co
        for _ in range(len(chans) - 1):
            L = (L + 1) // 2
        self.body = nn.Sequential(*layers)
        self.reduce = nn.Conv1d(c, head_c, 1)
        self.head = nn.Sequential(nn.Linear(head_c * L + 4, 256), nn.GELU(), nn.Linear(256, 4))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, xs, rs, res, dn):
        """xs, rs: 표준화한 실제·그린 곡선 (B,226), res: (x-R)/(10σ) (B,226), dn: 정규화한 지금 두께 (B,4)"""
        B, N = xs.shape
        grid = torch.linspace(-1, 1, N, device=xs.device).expand(B, N)
        h = self.body(self.stem(torch.stack([xs, rs, res, grid], 1)))
        z = torch.cat([nn.functional.gelu(self.reduce(h)).flatten(1), dn], 1)
        return self.head(z)                     # 정규화한 두께 보정량


ARCHS = ("fno", "mlp", "resnet", "cnn_tf")


def build_model(a):
    """a: 04_train.py 인자 dict. 예전 체크포인트에는 arch, k가 없으므로 fno, k=1로 본다.
    k>1이면 두께 후보 k세트(출력 4k개)를 낸다. 지금은 resnet만 지원."""
    arch, k = a.get("arch", "fno"), a.get("k", 1)
    if k > 1 and arch != "resnet":
        raise ValueError("후보 여러 개 출력은 resnet만 지원")
    if arch == "fno":
        return FNO1d(a["width"], a["modes"], a["layers"])
    if arch == "mlp":
        return MLP()
    if arch == "resnet":
        return ResNet1d(out=4 * k)
    if arch == "cnn_tf":
        return CNNTransformer()
    raise ValueError(arch)


def load_corrector(run_dir, ck, iters=None):
    """보정 신경망 체크포인트: 기본 모델 답에서 시작해 iters번 'TMM 곡선 그리기 -> 보정'을 반복한다."""
    from common import WORK, SIGMA, load_params
    from tmm_torch import TMMDecoder
    a = ck["args"]
    base = load_run(os.path.join(WORK, "runs", a["base"]))
    m = Corrector(); m.load_state_dict(ck["model"]); m.eval()
    xm, xs, dm, ds = ck["stats"]
    dec = TMMDecoder(load_params(), dtype=torch.float32)
    T = iters or a["iters"]

    @torch.no_grad()
    def predict(X, bs=2000):
        d = torch.as_tensor(base(X), dtype=torch.float32)
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        hist = [d.double().numpy()]
        for _ in range(T):
            out = []
            for s in range(0, len(Xt), bs):
                x, dc = Xt[s:s + bs], d[s:s + bs]
                R = dec(dc.clamp(0.5, 400))
                out.append(dc + m((x - xm) / xs, (R - xm) / xs, (x - R) / (10 * SIGMA), (dc - dm) / ds) * ds)
            d = torch.cat(out)
            hist.append(d.double().numpy())
        predict.history = hist                  # 반복마다의 두께 (보고용)
        return hist[-1]
    predict.last_candidates = None
    predict.history = None
    return predict


def load_run(run_dir, iters=None):
    """best.pt를 읽어 (X 넣으면 두께 nm (N,4)를 돌려주는 함수) 반환.
    후보 여러 개(k>1) 모델이면 후보마다 TMM으로 곡선을 그려 실제 곡선과 가장 잘 맞는 후보를 고른다
    (정답 라벨은 쓰지 않음). 고르기 전 후보 전체는 predict.last_candidates (N,k,4)에 남긴다.
    보정 신경망(arch=corrector)이면 load_corrector로 넘긴다 (iters로 반복 횟수를 바꿀 수 있음)."""
    from common import load_params, params_tuple, tmm
    ck = torch.load(os.path.join(run_dir, "best.pt"), weights_only=False, map_location="cpu")
    a = ck["args"]
    if a.get("arch") == "corrector":
        return load_corrector(run_dir, ck, iters)
    k = a.get("k", 1)
    m = build_model(a)
    m.load_state_dict(ck["model"]); m.eval()
    xm, xs, dm, ds = ck["stats"]
    par = params_tuple(load_params()) if k > 1 else None

    @torch.no_grad()
    def predict(X, bs=2000):
        out = [(m((torch.as_tensor(X[s:s + bs], dtype=torch.float32) - xm) / xs).view(-1, k, 4) * ds + dm)
               for s in range(0, len(X), bs)]
        C = torch.cat(out).double().numpy()                      # (N, k, 4)
        if k == 1:
            predict.last_candidates = None
            return C[:, 0]
        predict.last_candidates = C
        R = tmm(par, np.clip(C, 0.5, 400).reshape(-1, 4)).reshape(len(X), k, -1)
        best = ((R - np.asarray(X)[:, None]) ** 2).mean(2).argmin(1)
        return C[np.arange(len(X)), best]
    predict.last_candidates = None
    return predict
