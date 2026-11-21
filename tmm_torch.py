# 미분 가능한 TMM 디코더 (부록 A.6 기반). 실행하면 numpy TMM과 일치하는지 확인한다.
import sys, time
import numpy as np, torch
from common import load_params, params_tuple, tmm


class TMMDecoder(torch.nn.Module):
    """고정된 광학 상수로 두께 d(N,4, nm)에서 반사율 (N,226)을 계산."""

    def __init__(self, P, dtype=torch.float64):
        super().__init__()
        self.register_buffer("P", torch.tensor(P, dtype=dtype))

    def forward(self, d):
        cdt = torch.complex128 if self.P.dtype == torch.float64 else torch.complex64
        nH, kH, nL, kL, ns, ks, lam = self.P.T
        NH, NL, Ns = torch.complex(nH, -kH), torch.complex(nL, -kL), torch.complex(ns, -ks)
        d = d.to(self.P.dtype)
        B = torch.ones(d.shape[0], self.P.shape[0], dtype=cdt, device=d.device)
        C = B * Ns
        for j in (3, 2, 1, 0):
            Nj = NH if j in (0, 2) else NL
            dl = 2 * np.pi * Nj * d[:, j:j + 1] / lam
            c, s = torch.cos(dl), torch.sin(dl)
            B, C = c * B + 1j * s / Nj * C, 1j * s * Nj * B + c * C
        r = (B - C) / (B + C)
        return r.real ** 2 + r.imag ** 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    P = load_params()
    dec = TMMDecoder(P)
    d = (np.random.default_rng(0).integers(1, 31, (256, 4))) * 10.0
    d_t = torch.tensor(d, requires_grad=True)
    t = time.time()
    R = dec(d_t)
    R.sum().backward()
    print(f"256개 forward+backward {time.time() - t:.3f}s")
    gap = np.abs(R.detach().numpy() - tmm(params_tuple(P), d)).max()
    print("numpy와 최대 차이", gap)
    assert gap < 1e-9
