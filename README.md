# Reverse KL vs Forward KL · Multi-Modal Distribution Fitting Visualizer

[中文文档 (Chinese README)](./README.zh-CN.md)

🎮 **Live demo**: <https://qiangliu.net/reverse-vs-forward-kl-viz/>

An interactive demo that visualizes the **directional tendencies** of **reverse KL (q‖p)** and **forward KL (p‖q)** when fitting a multi-modal target distribution: **mode-seeking** vs **mode-covering**.

**Zero-dependency, browser-only** — a single `index.html` file runs the gradient descent, computes KL by numerical integration, and draws the density / loss curves on Canvas. No backend, no npm, no build tools. Open the file directly, or host it as a static page (GitHub Pages works).

> **Important**: This demo is meant to build **intuition** about directional tendencies. In real-world settings, the final fit behavior is **far more complex** than a one-line "mode-seeking / mode-covering" summary — it depends on hyperparameters, model capacity, and data distribution. See the [Complexity Note](#-complexity-note-final-behavior-is-not-determined-by-kl-direction-alone) below.

## Motivation

Primary references for understanding mode-seeking vs mode-covering:
- Eric Jang, *A Beginner's Guide to Variational Methods: Mean-Field Approximation* — <https://blog.evjang.com/2016/08/variational-bayes.html>
- Thinking Machines, *On-Policy Distillation* — <https://thinkingmachines.ai/blog/on-policy-distillation/>

Both articles converge on the same point: **the direction of the KL matters — the resulting behavior is very different**. This demo makes that concrete and interactive.

### Connection to On-Policy Distillation

The core argument in Thinking Machines' post is that LLM distillation should use **on-policy training + reverse KL**, rather than the traditional off-policy SFT (which is equivalent to a forward-KL MLE):

| Aspect | Forward KL / SFT | Reverse KL / On-policy distillation |
|---|---|---|
| Sample source | Teacher trajectories (offline) | Student's own rollouts (online) |
| Math form | $\mathbb{E}_{x\sim p_{teacher}}[-\log q_{student}(x)]$ | $\mathbb{E}_{x\sim q_{student}}[\log q - \log p_{teacher}]$ |
| Behavioral tendency | **Mode-covering** — cover all teacher modes, produce an "averaged" policy | **Mode-seeking** — commit to one optimal teacher mode, avoid mixed policies |
| Distribution shift | Severe (student ends up in states the teacher never visited) | None (data comes from the student itself) |

Paraphrased from their post:
> "reverse KL is 'mode-seeking' — it learns one specific behavior (the teacher's) rather than spreading its distribution across many suboptimal options."

**This demo's `K_p=3, K_q=1` configuration** is the one-dimensional scalar toy version of that idea:
- Teacher = p (a three-mode mixture; each mode can be imagined as a "valid behavior policy")
- Student = q (insufficient capacity — can only pick one mode)
- Reverse KL makes q commit to one peak → a **crisp, consistent** policy
- Forward KL makes q spread out to cover all three peaks → a **fuzzy, averaged** policy

> **But note**: this is a "loss-form tendency" analogy only. Whether reverse KL actually works in real LLM distillation depends on teacher quality, student capacity, rollout length, KL coefficient, and many other regularizers. **Do not extrapolate this 1D toy result directly** — it builds intuition, it doesn't prove that reverse KL is a silver bullet.

## What You'll See

Recommended configurations to feel the difference (results depend on the random seed — the table describes **typical tendencies**, not deterministic outcomes):

| Scenario | K_p | K_q | Common observation |
|---|---|---|---|
| ★ Classic contrast | 3 | 1 | Reverse KL collapses onto one peak; forward KL stretches into one wide Gaussian trying to cover all three |
| Under-parameterized | 5 | 2 | Reverse KL usually places its two components on two different peaks; forward KL often ends up "one wide, one narrow" or two wide ones — the exact split depends on initialization |
| Intermediate capacity | 6 | 5 | Interesting case: both KLs frequently **fall into local optima** — dropping a middle peak, or having multiple components collapse onto the same peak. See [Complexity Note](#-complexity-note-final-behavior-is-not-determined-by-kl-direction-alone) below |
| Sufficient capacity | 4 | 4 | Both KLs fit well; the visual difference essentially vanishes (control condition) |

**Tip**: set the seed to `42` to reproduce the table above; then try `7`, `123`, etc. to see how much variation exists for the same configuration. That variability is exactly what this demo aims to communicate.

## ⚠ Complexity Note: Final Behavior is Not Determined by KL Direction Alone

The "mode-seeking / mode-covering" framing above is a **directional tendency**, obtained by taking the problem to an extreme. In real experiments you'll see many phenomena that don't match the "textbook picture." **The final fit is jointly determined by**:

### 1. Model capacity (K_q)
- When q has enough capacity (K_q ≥ K_p), **both KLs fit accurately** and the "mode-seeking vs mode-covering" difference vanishes. Reverse KL also covers all modes here — because the optimum is q = p.
- The difference only manifests clearly when **q is under-parameterized (K_q < K_p)**.

### 2. Target distribution shape
- Intuitively: **larger mode separation and more uniform weights** → the mode-seeking / mode-covering visual difference becomes more dramatic
- Conversely, when modes overlap heavily or σ is large relative to spacing, p itself is already "blurred together" and the two KL optima look more similar
- **These are qualitative arguments from the definition of KL — verify empirically in the UI** (e.g., change σ in `TargetMixture` from 0.6 to 2.0 and see what happens)

### 3. Optimization hyperparameters
- **Learning rate**: too high → skip modes entirely; too low → stuck near the initialization
- **Initialization**: q's component means at init determine which p-peaks they get pulled toward. Same hyperparameters + different seeds → reverse KL can pick different modes
- **Iteration count**: too few steps and you may see an intermediate state; the characteristic mode-seeking / mode-covering shape only emerges after convergence

### 4. Local optima
- The KL loss over mixture models is **highly non-convex**. Two q-components collapsing onto the same p-peak (leaving other p-peaks orphaned) is a very common, mathematically **legitimate local optimum** — not a bug
- More capacity does not guarantee escape: even K_q = K_p = 5 often produces "3 components clumped on one peak, 2 modes missed"
- Classical GMM-EM tricks (k-means init, multi-restart) exist specifically to fight this problem

### 5. Sampling method (in real VI)
This demo computes KL by **numerical integration** — the gradient is exact and unbiased. In real VI/RL, KL can only be estimated by **Monte Carlo**, which introduces:
- High variance (reverse KL is especially prone to blow-ups when a q sample lands in a region where p has near-zero density)
- Estimator bias and instability, motivating extra regularization or constraints in many RL/VI algorithms (PPO clipping, TRPO trust regions, etc.)

### One-line summary
> **"Reverse KL is mode-seeking" is a qualitative statement about the loss function's preference — not a quantitative guarantee about the optimization outcome.**
> What you actually observe is the joint result of loss × model × data × optimizer × randomness.
> This demo shows the **easiest-to-reproduce extreme case**, not a universal rule.

**Try it empirically**: run K_p=5, K_q=3 a few dozen times (refresh + restart training each time). You'll see reverse KL sometimes pick the three left modes, sometimes the three right, and occasionally collapse two components onto the same peak. That randomness *is* part of the complexity.

## Directory Layout

```
kl-viz/
├── frontend/
│   └── index.html       # Everything: trainer + Canvas plots + MathJax formulas
├── README.md            # This file (English)
└── README.zh-CN.md      # Chinese version
```

That's it — one HTML file. The trainer (Gaussian mixture, hand-derived gradients, Adam optimizer, trapezoidal KL integration, seedable RNG) is ~250 lines of vanilla JS inside `<script>`.

## Mathematics

**Target distribution** p(x) — fixed Gaussian mixture:

$$p(x) = \sum_{k=1}^{K_p} w_k \, \mathcal{N}(x \mid \mu_k, \sigma_k^2)$$

In code, μ_k is placed uniformly on [-4·(K_p−1)/2, +4·(K_p−1)/2], σ_k = 0.6, w_k = 1/K_p.

**Variational distribution** q(x; θ) — trainable Gaussian mixture with parameters θ = {m_j, s_j, π_j}:

$$q(x;\theta) = \sum_{j=1}^{K_q} \pi_j \, \mathcal{N}(x \mid m_j, s_j^2)$$

**Both KLs** are computed by **numerical integration** (trapezoidal rule, 2000-point dense grid). Because the problem is 1D and both densities are analytic, no Monte Carlo sampling is needed; gradients w.r.t. θ are derived by hand and verified against finite differences:

$$D_\text{KL}(q\|p) = \int q(x) \log \tfrac{q(x)}{p(x)} \, dx \qquad D_\text{KL}(p\|q) = \int p(x) \log \tfrac{p(x)}{q(x)} \, dx$$

**Why not sample?** In real VI, p is typically only known up to normalization, so MC is unavoidable. Here both p and q are our own GMMs — numerical integration is more stable, `loss ≥ 0` holds strictly, and there's no log(0) blow-up. See the implementation notes at the end for a discussion.

## Quick Start

```bash
git clone <this-repo>
cd kl-viz
open frontend/index.html   # macOS
# or: xdg-open frontend/index.html   (Linux)
# or: start frontend/index.html      (Windows)
```

Or serve it as a static site:
```bash
cd kl-viz/frontend && python3 -m http.server 8765
# then open http://127.0.0.1:8765/
```

No pip, no venv, no torch — everything runs in the browser.

## Usage

1. Use the top sliders to pick K_p (1–6) and K_q (1–6)
2. Optional: fill in a **random seed** for reproducible runs (leave blank for fresh randomness each time)
3. Buttons:
   - **Reverse KL** — train a single reverse-KL fit
   - **Forward KL** — train a single forward-KL fit
   - **Compare Both** — run both in parallel via `requestAnimationFrame`, overlay red/green q-curves on the same plot
4. Watch the top panel (density curves) and the bottom panel (loss over steps)

## Implementation Highlights

- **Hand-derived analytic gradients**: no autograd framework needed. `∂q/∂m_j`, `∂q/∂log s_j`, `∂q/∂logit_j` are computed in closed form. Verified against finite differences across ~100 configurations; when the integration grid is wide enough to contain q's tails, relative error is ≤ 1e-8.
- **Trapezoidal integration on a dense grid** (2000 points spanning ±8 beyond p's range) — precise enough that `loss ≥ 0` holds strictly and there's no MC noise.
- **σ clamping**: σ is clamped to [0.05, 20] each forward pass to prevent numerical blow-ups.
- **Seedable RNG**: a mulberry32 PRNG lets you pin the initialization for reproducible runs, or leave blank to explore the local-optima landscape.
- **Zero dependencies**: no build step, no npm, no framework. MathJax is loaded from a CDN only for pretty formula rendering; you can strip it and the demo still works.

## References

- Eric Jang's post — <https://blog.evjang.com/2016/08/variational-bayes.html>
- Thinking Machines · On-Policy Distillation — <https://thinkingmachines.ai/blog/on-policy-distillation/>
- Bishop, *PRML*, Chapter 10 (variational inference, includes the canonical reverse KL vs forward KL comparison figure)
