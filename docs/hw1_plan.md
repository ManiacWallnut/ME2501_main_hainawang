# Assignment 1 — Exploration Plan (Dictionary Learning)

Working plan for HW1 after Part 0. Part 0 (a)–(d) and the MSP sanity check are
**done** (see "Status" below); this document fixes conventions, protocols,
expected outcomes and deliverables for the remaining parts, so each experiment
script can be written mechanically and the writeup assembled from JSONL logs.

Book references are to Wright & Ma, *High-Dimensional Data Analysis with
Low-Dimensional Models* (CUP 2022); the MSP algorithm is Zhai, Yang, Liao,
Wright & Ma, "Complete Dictionary Learning via ℓ⁴-Norm Maximization over the
Orthogonal Group", JMLR 21 (2020), Alg. 1/2.

---

## Status

| Item | State |
| --- | --- |
| 0(a) generator `data8014/dictlearn/generator.py` | done (BG codes Var=1, orthogonal / unit-norm-row dictionaries, noise) |
| 0(b) metrics `data8014/dictlearn/metrics.py` | done (eval loss via ISTA; recovery Eq. 3 **plus** one-to-one Hungarian matching and duplicate-match count) |
| 0(c) solvers/algorithms `data8014/dictlearn/{solvers,algorithms}.py` | done (ISTA/FISTA; MSP; warm-started alternating minimization, block scheme, η=1/(2‖A₊‖₂²), ν=1/(2‖Z⁺‖₂²) recomputed per block, unit-norm rows after A-step) |
| 0(d) ISTA warm-up `scripts/dl_part0_warmup_ista.py` | done: gap ≈ O(1/k) until entrywise support identified (t≈2273), then geometric ρ≈0.998, crossing below the Theorem-8.6 bound by ~10⁴× |
| MSP sanity check `scripts/dl_part0_sanity_msp.py` | done: 5/5 seeds, min_k s_k > 0.99 within 6–8 correctly one-based iterations (budget 30) |
| 0(e) orthogonality WLOG proof | derivation below; polished proof goes into the writeup |
| 1(a) `scripts/dl_part1a_msp_convergence.py` | coded + full 5-seed run: both metrics plateau by ~T=12; exponential transient fit beats power fit but R²≈0.8, so one simple law is not fully adequate |
| 1(b) `scripts/dl_part1b_msp_sample_size.py` | coded + full 5-seed run: all 45 runs stationary (max T=126); β_recovery=1.285 (R²=.936; per-seed 1.212±.192), β_eval=0.472 (R²=.907; per-seed .463±.077). Largest-N relative error: recovery 16.1%; eval raw 13.9% but **eval excess 207%**, so the measured-range fit does not extrapolate eval excess well |
| Part 2 `scripts/dl_part2_phase_transition.py` | coded + full 5-seed run: T* rises from 8±3 at θ=.2 to 88±17 at θ=.85; all runs stationary/no censoring; recovered fraction drops .988→.938→.488 at θ=.88/.90/.91 and reaches 0 by .95; eval loss changes smoothly and shows no matching discontinuity |
| Part 3 `scripts/dl_part3_mnist.py` | coded + full runs: PCA k=32 (80/90/99% need 44/87/331 PCs), whiten error 2.77e-7; T-axis final loss 10.189 (m=64) / 7.975 (m=128), learned/random ratios .640/.670, nnz .131/.074; N-axis β=.558/.485 with 1.22%/1.54% largest-N prediction error; 15/42 N-axis runs reached the permitted 1000-block cap and are disclosed; atoms are visually digit/stroke-like |
| Part 4 bonus `scripts/dl_part4_spectral_bias.py` | coded + full 5-seed run: empirical variances follow τ_k; altmin recovery range broadens with α (2–58, 1–79, 1–113; 80/78/72 of 80 recovered); MSP at α=1 recovers all 80 atom-seed pairs in 1–6 iterations with little energy ordering |
| LaTeX writeup | 13-page draft at `writeup/hw1_writeup.tex`; Tectonic compile succeeds with no TeX warnings; student identity, repository URL, and personal verification remain |

## Conventions for all experiments (fixed in Part 0, reused throughout)

- **Metrics.** Recovery is reported in the primary "as-defined" form (Eq. 3,
  per-atom max |corr|), and we explicitly report the excess duplicate matches
  (`dup_matches = m − number of distinct argmax partners`); the one-to-one
  Hungarian matching mean (`mean_s_onetoone`) is reported alongside. Eval loss
  always solves (6) using ISTA at **λ = 0.6** unless the part says otherwise.
  In the complete regime every MSP iterate is orthogonal, so ISTA initialized
  at zero reaches the exact LASSO solution in **one step**:
  `Z = soft(XÂᵀ, λ/2)`; use one ISTA step there. In the overcomplete regime use
  a fixed documented iteration budget (or an objective tolerance) large enough
  for validation inference. The metric and inference budget are named in every
  reported number.
- **Errors** (Eq. 5): `eval_loss` and `recovery_error = 1 − mean_k s_k`.
- **Seeds:** ≥ 3 per point; report mean ± std (band on plots).
- **Artifacts:** each script writes a JSONL log under `outputs/logs/` (gitignored)
  and figures under `outputs/figures/` (tracked); plots are regenerated from
  logs, never from memory.
- **Environment:** `.venv` (uv-managed CPython 3.12.13 + numpy/scipy/matplotlib
  from PyPI). The system Anaconda numpy has a broken BLAS/LAPACK on this
  machine (delay-load DLL failure) — do not use it. The dictlearn stack is
  numpy-only; torch/torchvision appear only in Part 3 (MNIST), with a stdlib
  IDX-download fallback so the venv stays light.
- **float32** only in Part 3 as instructed; Parts 0–2 stay float64.

---

## Part 1 — Optimization convergence and sample-size scaling

Settings: D = 16, θ = 0.1, complete regime, σ = 0, λ = 0.6 for eval loss.

### 1(a) T-axis (optimization convergence)

- Script `scripts/dl_part1a_msp_convergence.py`.
- N = 8000; default T = 100 (CLI-overridable after inspecting the plateau),
  seeds {0,1,2,3,4}; use one fixed validation set and one independent random
  orthogonal initialization per seed; include T = 0 and record both metrics at
  every iteration. One ISTA step computes each orthogonal-dictionary eval loss
  exactly, so frequent evaluation is cheap.
- Plots: the **raw** eval loss and recovery error vs T, each on semilogy and
  log-log axes (2×2 panel, mean ± std across seeds).
- Quantitative diagnosis must account for finite-sample / LASSO floors. Estimate
  each metric's optimization floor from its final stable tail, fit the positive
  transient `e_T − e_floor` to both `T^{−α}` and `ρ^T`, and compare transformed
  R² on the same automatically selected pre-floor points. Also report fits of
  the raw metric if it does not have a visible floor. Never classify a rate
  from the final numerical-floor points.
- **Hypothesis, not assumed conclusion.** Population MSP has local cubic
  convergence, but a finite-sample run may show a short superlinear transient
  followed by a statistical fixed-point floor; therefore either model in (7)
  may be inadequate. Eval loss additionally has the λ-induced shrinkage floor,
  so it need not share the recovery-error transient.
- Deliverables: `dl_part1a_msp_convergence.png`, JSON summary with floor,
  α/ρ, fit window, number of fit points and R² for each metric.

### 1(b) N-axis (sample-size scaling)

- Script `scripts/dl_part1b_msp_sample_size.py`.
- N ∈ {100, 178, 316, 562, 1000, 1780, 3160, 5620, 10000}: exactly a factor
  100 (> the required six points). For each seed, generate N_max = 10000 once
  and use prefixes, while holding A*, validation data and A₀ fixed across N.
  This isolates sample count instead of confounding it with a new dictionary,
  validation set or initialization.
- "Optimization converged" is an **optimization stationarity** condition, not
  closeness to ground truth (small-N statistical error need not vanish): stop
  after the one-to-one signed/permutation-invariant alignment between A_t and
  A_{t−1} changes by < 1e-12 for five consecutive iterations, after at least
  20 iterations; cap at 5000 and report capped runs.
- Eval-loss floor E_∞: for each seed solve (6) once with A* on the same fixed
  validation set (one exact ISTA step). Record raw eval loss and fit the excess
  `eval_loss − E_∞`. Do not silently clip negative finite-validation excesses;
  flag them and omit only invalid points from a log-space fit.
- Fit (8) to the mean curve at the smallest six N values (the smallest 2/3),
  extend it to all N, and predict N = 10000. Report β, absolute and relative
  extrapolation errors; for eval loss report errors on both the raw loss and
  the fitted quantity `E−E∞` so the floor cannot mask a poor extrapolation.
  Additionally fit per seed and report exponent mean ± std.
  Recovery uses E_∞ = 0. Plot mean ± std on log-log axes.
- Deliverables: `dl_part1b_sample_size.png`, JSON summary with β, fit points,
  prediction and extrapolation errors for both metrics.

---

## Part 2 — Phase transition in identifiability

Complete regime, D = 16, σ = 0, N = 8000, ≥ 3 seeds per point.

- Script `scripts/dl_part2_phase_transition.py` (one script, three sub-experiments
  selected by flags; shared sweep harness).
- **2.1 cost of density.** θ ∈ {0.2, 0.4, 0.6, 0.7, 0.8, 0.85}; T\*(θ) = first
  iteration with ≥ 95% of atoms at s_k > 0.9. Since D = 16, this means all 16
  atoms (15/16 = 93.75%). Start with cap 20000, benchmark rather than assuming
  1 ms/iteration, checkpoint JSONL incrementally, and record censored runs as
  `T* > cap` instead of treating the cap as an observed recovery time. Plot
  per-seed points, mean and spread; do not pre-label the growth law.
- **2.2 the transition.** θ ∈ {0.88, 0.90, 0.91, 0.92, 0.93, 0.95, 1.0}; use
  the same fixed large budget across θ, while also recording iterate/objective
  stationarity so optimization failure can be distinguished from an empirical
  recovery failure. Report the final fraction of atoms with s_k > 0.9 (mean ±
  std and per-seed points). At θ = 1 the recovered orientation is arbitrary.
- **2.3 can the loss see it?** For every θ in the **union of 2.1 and 2.2**,
  report held-out reconstruction loss of the corresponding final dictionary,
  using one exact ISTA step. Plot recovery and loss together; determine from
  the data, rather than assume, whether the loss reveals the transition.
- **2.4 why there** (derivation for the writeup):
  - (a) E[z²] = 1, E[z⁴] = 3/θ ⇒ κ(θ) = 3(1−θ)/θ **> 0** for θ < 1;
    evaluate this at the empirically observed transition, not a preselected
    θ. κ(1) = 0.
  - (b) At θ = 1, each code row is N(0,I). For any orthogonal R, define
    `Z' = ZRᵀ` and `A' = RA`; then `Z'A' = ZA`, while Z' has the same iid
    Gaussian law as Z. Infinitely many row-rotated dictionaries therefore
    induce exactly the same data distribution.
  - (c) BG codes for θ < 1 are **leptokurtic** (positive excess kurtosis), not
    platykurtic. MSP's ℓ⁴ objective exploits this non-Gaussian fourth-moment
    anisotropy: signed-permutation alignment exposes sparse high-amplitude
    coordinates. As θ → 1, κ → 0 and the Gaussian law / population objective
    becomes rotationally invariant, so the dictionary-identifying signal
    vanishes. Finite-N fluctuations then dominate near the empirical
    transition.
- Deliverables: `dl_part2_Tstar_vs_theta.png`, `dl_part2_transition.png`
  (recovery fraction and eval loss vs θ, two panels), numeric tables.

---

## Part 3 — Real data (MNIST)

- Script `scripts/dl_part3_mnist.py` (+ small loader in
  `data8014/dictlearn/mnist.py`: try torchvision, else direct IDX download
  from the ossci-datasets mirror; numpy-only pipeline).
- **(a) Spectrum + projection.** 60k train / fixed 2k test, flattened 784-d;
  eigenvalue spectrum of Cov(x) on log axes; variance explained at 80/90/99%;
  project to top k = 32 PCA directions, then **whiten** (preconditioning per
  Remark 2.8 + Part 0(e)): X ← X Σ⁻¹/²; verify ‖Cov(X_precond) − I‖_F / ‖I‖_F ≈ 0
  and report it. From here D = k = 32.
- **(b) Alternating minimization.** m = 2k = 64 and m = 4k = 128, λ = 2,
  float32; eval loss (5) on the fixed 2k test set with ISTA (λ = 2), ≥ 3 seeds.
  - T-axis: N = 12000, 400 blocks, record held-out loss + training objective
    every 5 blocks; semilogy + log-log plots; compare with Part 1(a) (expect
    qualitatively similar early decay; no true convergence floor knowledge).
  - N-axis: per seed a fixed random image order; N ∈ {200, 400, 800, 1600,
    3200, 6400, 12800}; stop when relative training-objective change over the
    preceding 50 blocks is < 1e-5 (cap 1000). Fit
    `E(N)=E_∞+cN^{−β}` with constrained nonlinear least squares
    (`E_∞≥0, c>0, β>0`); this is **not** log-linear when E_∞ is free. Use the
    smallest five points (smallest 2/3) for the fit and predict N=12800; report
    absolute/relative prediction errors and parameter uncertainty.
  - Throughout: random-dictionary baseline (unit-norm Gaussian rows) with the
    same k,m and the same inference budget; define and report the ratio as
    `learned held-out loss / random held-out loss` (<1 is better), plus the
    fraction of nonzero inferred code entries at λ=2. T-axis plots include the
    random baseline as a horizontal reference.
  - Budget: benchmark one block before launching the full grid. Persist every
    checkpoint so the run is resumable; do not reduce the required ≥3 seeds.
    Float32 must propagate through code arrays (not only X and A).
- **(c) Atoms.** With centered projected data
  `Y=(X−μ)V_k`, whitening `Y_w=Y diag(λ_k^{-1/2})`, and learned whitened atom
  row a_w, map the atom direction back as
  `a_pixel = a_w diag(sqrt(λ_k)) V_kᵀ`. **Do not add μ to each atom**: μ is the
  data offset, not an atom direction. Normalize each displayed atom separately
  for contrast; render grids for m=64 and m=128 and interpret only after
  inspecting the images.
- Deliverables: `dl_part3_spectrum.png`, `dl_part3_taxis.png`,
  `dl_part3_naxis.png`, `dl_part3_atoms_m64.png`, `dl_part3_atoms_m128.png`,
  tables (β, ratios, nnz fractions).

---

## Part 4 — Spectral bias (bonus)

- Extend the generator (`per_atom_energy=True`): p_k = c k^{−α}, v_k = 1,
  no rescaling; report empirical Var(z_k) per k (must vary).
- Script `scripts/dl_part4_spectral_bias.py`: complete regime, c = 0.45,
  D = m = 16, N = 12000, σ = 0.01, λ = 0.6, T = 120 outer × L = 30 inner;
  recovery time t_k = first outer iteration with s_k > 0.8 (else NaN).
- 4.1 derivations go to the writeup (E[z] = 0; Cov(z) = diag(τ); Cov(x) =
  Aᵀdiag(τ)A + σ²I; eigendecomposition of Cov(x) in the orthogonal regime;
  v_k = 1/p_k ⇒ τ_k ≡ 1 ⇒ Part 0 generator is spectrally blind by design).
- 4.2 alternating minimization: scatter t_k vs τ_k, log-log, α ∈ {0.75, 1,
  1.25}, 3 seeds; unrecovered atoms drawn above the budget line. Expect
  negative-slope ordering strengthening with α; report seed spread.
- 4.3 MSP at α = 1 side-by-side on **identical generated data and A₀**. Do not
  assume the ordering persists: MSP's polar projection updates and couples an
  entire orthogonal basis at once (and can have a superlinear local phase),
  whereas alternating minimization learns through energy-weighted inferred
  codes. The experiment decides whether this joint orthogonality constraint
  suppresses, preserves or strengthens the per-atom ordering.
- Deliverables: `dl_part4_tk_vs_tau.png` (2×3 panels: 3 α × 2 algorithms),
  range-of-t_k table.

---

## Part 0(e) derivation (for the writeup; recorded here so it is not lost)

**Claim.** With z generated as in Part 0(a) (iid entries, Var = 1) and
E = 0, assuming A invertible, it suffices to study orthogonal dictionaries.

*Proof.* Let x = Aᵀz (row-vector convention: rows of X). The population
covariance is Cov(x) = Aᵀ E[zzᵀ] A = Aᵀ A. Apply the linear preconditioner
W = (AᵀA)^{−1/2} to the data: XW = Z A (AᵀA)^{−1/2} = Z Q with
Q = A(AᵀA)^{−1/2}. Then QQᵀ = A(AᵀA)^{−1}Aᵀ = A A^{−1}A^{−T} Aᵀ = I, so Q is
orthogonal (m = D), and — crucially — the codes are *unchanged*: the same Z
with the same iid BG law. Hence every invertible-dictionary instance is
mapped by a fixed linear transformation of the data to an orthogonal-dictionary
instance with identical codes; conversely any estimator for the orthogonal
problem pulls back through W^{−1}. ∎

(Practical corollary used in Part 3(a): whiten by the *empirical* covariance
factor Σ̂^{−1/2} — the same reduction without knowing A.)

Numerical micro-check to include in the writeup appendix: generate an
invertible (non-orthogonal) A, run MSP on whitened data, confirm recovery of
Q = A(AᵀA)^{−1/2} up to signed permutation.

---

## Milestones

1. **M1 (done)** — Part 0 + sanity check.
2. **M2 (done)** — Part 1 shared fit/stopping utilities, tests, full 5-seed
   runs and figures.
3. **M3 (done)** — Part 2 full 5-seed sweep, transition analysis and figures;
   all transition runs reached stationarity before the 20000-iteration cap.
4. **M4 (done)** — Part 3 MNIST loader, whitening, resumable T/N-axis runs,
   random baselines, scaling fits and atom figures.
5. **M5 (done)** — Part 4 bonus generator, alternating/MSP runs and plots.
6. **M6 (draft complete)** — LaTeX writeup compiled to PDF; fill three red TODOs and personally verify before submission (`writeup/writeup.tex` skeleton exists): intro,
   methods (implementations + conventions), per-part figures + explanations,
   0(e)/2.4/4.1 derivations, AI-usage disclosure; compile, push, submit with
   commit SHA.

## Risks / mitigations

- **MSP stall at high θ** (2.1–2.2): cap budget, report honestly; the cap
  itself is the interesting data point near the transition.
- **Power-law fits may be poor** (1b, 3b): report fit diagnostics (R²,
  residuals); where MSP's super-algebraic rate contaminates the asymptote,
  note that the model (8) describes the *statistical* regime only after
  optimization has fully converged.
- **float32 accumulation in Part 3**: use pairwise/BLAS-default accumulation,
  keep λ-solves in float32 but compare one float64 run for sanity.
- **MNIST download blocked**: fallback mirror + committed SHA1 check; worst
  case switch to CIFAR10 (first channel) as the assignment allows.
- **Seed variance near the transition**: report std and per-seed histograms,
  never only means.
