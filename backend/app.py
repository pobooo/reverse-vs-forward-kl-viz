"""
Reverse / Forward KL visualization backend.

Target distribution p(x): fixed Gaussian mixture with K_p modes.
Variational distribution q(x; theta): trainable Gaussian mixture with K_q components.

Reverse KL:  D_KL(q || p) = E_q[log q - log p]     -> mode-seeking
Forward KL:  D_KL(p || q) = E_p[log p - log q]     -> mode-covering (mean-seeking)

Both losses are estimated by Monte Carlo samples so gradients flow through q
via the reparameterization trick.
"""
from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

LOG_2PI = math.log(2.0 * math.pi)


def gaussian_log_prob(x: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """log N(x | mu, sigma^2). Broadcasts x[..., None] against components."""
    sigma2 = torch.exp(2.0 * log_sigma)
    return -0.5 * (LOG_2PI + 2.0 * log_sigma + (x - mu) ** 2 / sigma2)


def mixture_log_prob(x: torch.Tensor, mus: torch.Tensor, log_sigmas: torch.Tensor,
                     log_weights: torch.Tensor) -> torch.Tensor:
    """log sum_k w_k N(x | mu_k, sigma_k^2). x: (N,), returns (N,)."""
    # (N, K)
    comp = gaussian_log_prob(x.unsqueeze(-1), mus, log_sigmas) + log_weights
    return torch.logsumexp(comp, dim=-1)


class TargetMixture:
    """Fixed target Gaussian mixture p(x) with K_p modes spaced along the real line."""

    def __init__(self, n_modes: int, device: str = "cpu"):
        self.n_modes = n_modes
        self.device = device
        # Evenly spaced modes, unit-ish variance, equal weights.
        if n_modes == 1:
            centers = np.array([0.0])
        else:
            centers = np.linspace(-4.0 * (n_modes - 1) / 2, 4.0 * (n_modes - 1) / 2, n_modes)
        self.mus = torch.tensor(centers, dtype=torch.float32, device=device)
        self.log_sigmas = torch.full((n_modes,), math.log(0.6), dtype=torch.float32, device=device)
        self.log_weights = torch.full((n_modes,), -math.log(n_modes), dtype=torch.float32, device=device)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return mixture_log_prob(x, self.mus, self.log_sigmas, self.log_weights)

    def sample(self, n: int) -> torch.Tensor:
        k = torch.multinomial(torch.exp(self.log_weights), n, replacement=True)
        eps = torch.randn(n, device=self.device)
        return self.mus[k] + torch.exp(self.log_sigmas[k]) * eps

    def to_dict(self):
        return {
            "mus": self.mus.detach().cpu().tolist(),
            "sigmas": torch.exp(self.log_sigmas).detach().cpu().tolist(),
            "weights": torch.exp(self.log_weights).detach().cpu().tolist(),
        }


class VariationalMixture(nn.Module):
    """Trainable Gaussian mixture q(x; theta) with K_q components.

    Sampling uses the Gumbel-Softmax reparameterization so that gradients flow
    through the mixture weights as well as (mu, sigma).
    """

    def __init__(self, n_components: int, init_range: float = 3.0, device: str = "cpu"):
        super().__init__()
        self.n_components = n_components
        # Initialize means spread across init_range so components don't all
        # start inside a single mode of p.
        if n_components > 1:
            init_mus = torch.linspace(-init_range, init_range, n_components)
        else:
            init_mus = torch.zeros(1)
        # Jitter proportional to the spacing so we break symmetry without
        # collapsing neighbors onto each other.
        spacing = (2 * init_range) / max(n_components - 1, 1)
        init_mus = init_mus + 0.15 * spacing * torch.randn(n_components)
        self.mus = nn.Parameter(init_mus.to(device))
        self.log_sigmas = nn.Parameter(torch.full((n_components,), math.log(1.0), device=device))
        self.logits = nn.Parameter(torch.zeros(n_components, device=device))

    @property
    def log_weights(self) -> torch.Tensor:
        return F.log_softmax(self.logits, dim=-1)

    @property
    def _clamped_log_sigmas(self) -> torch.Tensor:
        # Keep sigma in [~0.05, ~20] to avoid numerical blow-ups.
        return self.log_sigmas.clamp(min=math.log(0.05), max=math.log(20.0))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return mixture_log_prob(x, self.mus, self._clamped_log_sigmas, self.log_weights)

    def rsample(self, n: int, tau: float = 0.5) -> torch.Tensor:
        """Reparameterized sample from q.

        For the mixture case, we use *hard* Gumbel-Softmax (straight-through) so
        that the forward pass draws from a single component (matching the true
        mixture density used in log_prob), while gradients still flow through
        the mixture weights.
        """
        log_sig = self._clamped_log_sigmas
        if self.n_components == 1:
            eps = torch.randn(n, device=self.mus.device)
            return self.mus[0] + torch.exp(log_sig[0]) * eps
        # Hard Gumbel-Softmax: forward is one-hot, backward uses soft gradient.
        onehot = F.gumbel_softmax(self.logits.expand(n, -1), tau=tau, hard=True)
        eps = torch.randn(n, self.n_components, device=self.mus.device)
        xk = self.mus + torch.exp(log_sig) * eps
        return (onehot * xk).sum(dim=-1)

    def to_dict(self):
        return {
            "mus": self.mus.detach().cpu().tolist(),
            "sigmas": torch.exp(self._clamped_log_sigmas).detach().cpu().tolist(),
            "weights": torch.exp(self.log_weights).detach().cpu().tolist(),
        }


# ---------------------------------------------------------------------------
# KL estimators
# ---------------------------------------------------------------------------

def reverse_kl_loss(q: VariationalMixture, p: TargetMixture, grid: torch.Tensor) -> torch.Tensor:
    """D_KL(q || p) = integral q(x) [log q(x) - log p(x)] dx, via trapezoidal rule.

    Since both p and q are known GMMs with tractable densities, we can compute
    the KL by numerical integration on a dense grid. No sampling noise, no
    negative estimates due to Monte Carlo variance.
    """
    log_q = q.log_prob(grid)
    log_p = p.log_prob(grid)
    q_pdf = torch.exp(log_q)
    integrand = q_pdf * (log_q - log_p)
    # Trapezoidal integration
    dx = grid[1] - grid[0]
    return torch.trapezoid(integrand, dx=dx)


def forward_kl_loss(q: VariationalMixture, p: TargetMixture, grid: torch.Tensor) -> torch.Tensor:
    """D_KL(p || q) = integral p(x) [log p(x) - log q(x)] dx, via trapezoidal rule."""
    log_q = q.log_prob(grid)
    log_p = p.log_prob(grid)
    p_pdf = torch.exp(log_p)
    integrand = p_pdf * (log_p - log_q)
    dx = grid[1] - grid[0]
    return torch.trapezoid(integrand, dx=dx)


# ---------------------------------------------------------------------------
# Training loop with streaming
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    kl_type: str            # "reverse" or "forward"
    target_modes: int
    q_components: int
    steps: int = 400
    lr: float = 0.05
    n_samples: int = 512
    send_every: int = 2     # push a frame every N steps


def build_grid(p: TargetMixture, extra: float = 3.0, n: int = 400) -> torch.Tensor:
    lo = float(p.mus.min()) - extra
    hi = float(p.mus.max()) + extra
    return torch.linspace(lo, hi, n)


async def run_training(ws: WebSocket, cfg: TrainConfig):
    device = "cpu"
    p = TargetMixture(cfg.target_modes, device=device)
    # Match q's init range to p's spread + some margin so all components start
    # inside the region where p has mass.
    p_extent = float(p.mus.abs().max()) + 2.0
    q = VariationalMixture(cfg.q_components, init_range=p_extent, device=device)
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)

    # Plot grid (sparser, for the frontend curve).
    grid = build_grid(p)
    grid_list = grid.cpu().tolist()
    p_pdf = torch.exp(p.log_prob(grid)).detach().cpu().tolist()

    # Integration grid: denser so the trapezoidal KL is close to the true value.
    lo = float(p.mus.min()) - 8.0
    hi = float(p.mus.max()) + 8.0
    int_grid = torch.linspace(lo, hi, 2000, device=device)

    await ws.send_json({
        "type": "init",
        "grid": grid_list,
        "p_pdf": p_pdf,
        "target": p.to_dict(),
        "config": {
            "kl_type": cfg.kl_type,
            "target_modes": cfg.target_modes,
            "q_components": cfg.q_components,
            "steps": cfg.steps,
            "lr": cfg.lr,
        },
    })

    loss_fn = reverse_kl_loss if cfg.kl_type == "reverse" else forward_kl_loss

    for step in range(cfg.steps + 1):
        opt.zero_grad()
        loss = loss_fn(q, p, int_grid)
        loss.backward()
        # Clip for numerical safety.
        torch.nn.utils.clip_grad_norm_(q.parameters(), 5.0)
        if step > 0:  # step 0 = initial state, no update yet
            opt.step()

        if step % cfg.send_every == 0 or step == cfg.steps:
            with torch.no_grad():
                q_pdf = torch.exp(q.log_prob(grid)).cpu().tolist()
            try:
                await ws.send_json({
                    "type": "step",
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "q_pdf": q_pdf,
                    "q": q.to_dict(),
                })
            except (WebSocketDisconnect, RuntimeError):
                return
            # Yield to event loop so the client actually sees frames.
            await asyncio.sleep(0.01)

    await ws.send_json({"type": "done"})


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="KL Divergence Visualizer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/train")
async def ws_train(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            payload = json.loads(msg)
            if payload.get("action") == "start":
                cfg = TrainConfig(
                    kl_type=payload.get("kl_type", "reverse"),
                    target_modes=int(payload.get("target_modes", 3)),
                    q_components=int(payload.get("q_components", 1)),
                    steps=int(payload.get("steps", 400)),
                    lr=float(payload.get("lr", 0.05)),
                    n_samples=int(payload.get("n_samples", 512)),
                    send_every=int(payload.get("send_every", 2)),
                )
                await run_training(ws, cfg)
            elif payload.get("action") == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        return


# Serve the static frontend at "/"
import pathlib
FRONTEND_DIR = pathlib.Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
