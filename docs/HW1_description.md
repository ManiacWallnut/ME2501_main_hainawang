# Assignment 1: Exploring Dictionary Learning

**Data 8014 — Fall 2026**

---

Given data $\mathbf{X} \in \mathbb{R}^{N\times D}$ whose rows $\mathbf{x}_1^\top,\dots,\mathbf{x}_N^\top$ are the samples, we seek a dictionary $\mathbf{A} \in \mathbb{R}^{m\times D}$ whose $m$ rows are the *atoms*, and a sparse code matrix $\mathbf{Z} \in \mathbb{R}^{N\times m}$ whose rows are the codes $\mathbf{z}_1^\top,\dots,\mathbf{z}_N^\top$, such that

$$\mathbf{X} = \mathbf{Z}\mathbf{A} + \mathbf{E}, \qquad \|\mathbf{z}_i\|_0 \ll m \quad \forall i, \tag{1}$$

where $\mathbf{E}$ is small noise. The dictionary is *complete* when $m = D$ and *overcomplete* when $m \gg D$. In the complete case we further assume $\mathbf{A}$ is orthogonal.

## What you will get out of this assignment

We expect that after you finish this homework, you will understand complete and overcomplete dictionary learning better, including the convergence behavior of different algorithms and how the problem scales with data. You will see that sparsity plays a central role in determining both the difficulty and identifiability of the dictionary learning problem: as the codes become progressively denser, both can change substantially, and the reconstruction error will fail to inform you in this regard. You will also find that not all atoms are learned at equal pace, a phenomenon that carries over and becomes an important concept for diffusion models, which we will study in the next homework. Finally, we hope you can bring together what you have learned to apply dictionary learning to real-world data.

**Resource budget.** $D \le 64$, $m \le 256$, $N \le 20000$. A GPU is not required: a laptop should be enough.

---

## Part 0: Setup

You need three things before starting. Please keep them in the semester-long codebase.

### (a) A data generator.

Generate the sparse codes following Bernoulli–Gaussian as mentioned in Section 2.2.2 of the book:

$$z_k \sim \mathrm{Bern}(\theta)\cdot\mathcal{N}(0,\,1/\theta), \tag{2}$$

independently across $k$, so that $\mathrm{Var}(z_k)=1$ and $\mathbb{E}\|\mathbf{z}\|_0=m\theta$. Generate the dictionary $\mathbf{A}$ as follows: in the complete case, take the orthogonal $\mathbf{Q}$ factor from the QR decomposition of a Gaussian matrix; in the overcomplete case, draw its rows independently from a Gaussian distribution and normalize each row to have unit $\ell^2$ norm. Assume the noise $\mathbf{E}$ follows an i.i.d. Gaussian with noise level $\sigma$. Expose $D$, $m$, $N$, $\theta$, noise level $\sigma$, a random seed, and a flag indicating either the complete ($m=D$, orthogonal) or overcomplete ($m>D$, unit-norm rows) setting.

### (b) Two metrics.

- **Eval loss**: $\|\mathbf{X}-\hat{\mathbf{Z}}\hat{\mathbf{A}}\|_F^2/N_{\mathrm{val}}$ on validation samples from the same ground-truth dictionary.
- **Recovery**: Since dictionaries are identifiable only up to permutation and sign, compare true and recovered atoms using absolute correlation:

  $$s_k \doteq \max_j \bigl|\langle \mathbf{a}_k, \hat{\mathbf{a}}_j\rangle\bigr|, \tag{3}$$

  where all atoms are normalized to unit norm. Report the mean of $s_k$ and the fraction of atoms satisfying $s_k>0.9$. Note that this metric allows multiple true atoms to match the same recovered atom, which would sometimes favor a collapsed dictionary. Either report how often such duplicate matches occur or enforce a one-to-one matching between true and recovered atoms. Clearly state which approach you use.

### (c) Algorithms.

First implement an ISTA solver (Eq. 2.3.9) for the case where $\mathbf{A}$ is known. Then implement one dictionary-learning algorithm for each regime:

- **Complete**: the MSP algorithm (Eq. 2.2.18).
- **Overcomplete**: alternating minimization (Eqs. 2.3.18–2.3.21), with ISTA as the inner solver. We make a few modifications to the procedure in the book.

  Use the "block" scheme with $L=30$ inner ISTA steps and one dictionary update ($T=1$) per block, and report the iteration axis in units of blocks. In addition, **warm-start the codes**: instead of initializing $\tilde{\mathbf{Z}}^1=\mathbf{0}$ at the beginning of every block as in Eq. 2.3.18, initialize $\tilde{\mathbf{Z}}$ from the codes obtained in the previous block and perform $L$ additional ISTA steps. Since $\mathbf{A}_+$ typically changes only slightly between consecutive blocks, the previous codes provide a good initialization for the new LASSO subproblem.

  Use step sizes

  $$\eta=\frac{1}{2\|\mathbf{A}_+\|_2^2}, \qquad \nu=\frac{1}{2\|\mathbf{Z}^+\|_2^2},$$

  and recompute both at every block.

### (d) Warm-up: convergence of ISTA.

Work in the overcomplete regime with $N=2000$, $D=16$, $m=32$, $\theta=0.1$, $\lambda=0.6$ and $\sigma=0$. Read Chapter 8.2.2–8.2.3 of Wright and Ma to understand the convergence rate of ISTA for the inner LASSO problem in the alternating minimization algorithm:

$$\min_{\tilde{\mathbf{Z}}}\ \|\mathbf{X}-\tilde{\mathbf{Z}}\tilde{\mathbf{A}}\|_F^2+\lambda\|\tilde{\mathbf{Z}}\|_1, \tag{4}$$

where $\tilde{\mathbf{A}}$ is fixed to the ground-truth dictionary $\mathbf{A}$.

Define $f(\tilde{\mathbf{Z}}) = \|\mathbf{X}-\tilde{\mathbf{Z}}\tilde{\mathbf{A}}\|_F^2 + \lambda\|\tilde{\mathbf{Z}}\|_1$ and $f^\star=\min_{\tilde{\mathbf{Z}}} f(\tilde{\mathbf{Z}})$. Then verify the result numerically by plotting the objective gap $f(\tilde{\mathbf{Z}}_t)-f^\star$ together with the theoretical bound. Does the observed decay match the theoretical rate, or does it eventually become faster? If it becomes faster, explain what structural property of the LASSO solution is not captured by the worst-case convex bound. *Hint: watch the support of $\tilde{\mathbf{Z}}_t$.*

### (e) Warm-up: why orthogonality costs no generality.

Above, in the complete case, we assumed that $\mathbf{A}$ is orthogonal. Show that this assumption is without loss of generality. Assume that $\mathbf{z}$ is generated as in part (a), that $\mathbf{A}$ is invertible, and that $\mathbf{E}=\mathbf{0}$.

> **MSP Sanity check.** With $\sigma=0$, $\theta=0.2$, $D=16$, $N=8000$, complete regime, MSP should reach $s_k > 0.99$ for essentially every atom within about 30 iterations.

---

## Part 1: Optimization convergence and sample-size scaling

**Notation and conventions.** Write $e_t$ generically for either error metric after $t$ (outer) iterations. Throughout this homework, we track two notions of error:

$$\text{eval loss} \quad \frac{1}{N_{\mathrm{val}}}\|\mathbf{X}_{\mathrm{val}}-\hat{\mathbf{Z}}_{\mathrm{val}}\hat{\mathbf{A}}\|_F^2, \qquad \text{recovery error} \quad 1-\frac1m\sum_k s_k. \tag{5}$$

For the eval loss, the validation codes should be inferred using the learned dictionary by solving

$$\hat{\mathbf{Z}}_{\mathrm{val}} \approx \arg\min_{\mathbf{Z}} \|\mathbf{X}_{\mathrm{val}}-\mathbf{Z}\hat{\mathbf{A}}\|_F^2 +\lambda\|\mathbf{Z}\|_1, \tag{6}$$

using ISTA. The two metrics need not converge at the same rate, so whenever you report an error, clearly state which metric you are using.

This part studies two different axes:

- varying the number of optimization iterations $T$ measures *optimization convergence*;
- varying the number of samples $N$ after optimization has converged measures *sample-size scaling*.

Unless stated otherwise, use $D=16$ and $\theta=0.1$, and use $\lambda=0.6$ when computing the held-out eval loss in (6). For every reported quantity, run at least three random seeds and report the mean and standard deviation.

### (a) Optimization convergence: the T-axis.

Set $\sigma=0$ and $N=8000$. Run MSP and measure both metrics in (5) as functions of the number of fitting iterations $T$.

Plot each curve on both semilogy and log–log axes. Based on the plots and a quantitative fit, characterize the observed convergence behavior. In particular, determine whether the decay is more consistent with

$$e_T \propto T^{-\alpha} \qquad\text{or}\qquad e_T \propto \rho^T, \quad 0<\rho<1, \tag{7}$$

or whether neither description is adequate. Estimate it using a sufficiently long run.

Do the eval loss and recovery error exhibit the same convergence behavior? If not, describe the difference.

### (b) Sample-size scaling: the N-axis.

We now separate optimization error from statistical error. Set $\sigma=0$, vary the sample count $N$ across at least two orders of magnitude with at least six points (for example, $500$ to $10{,}000$), and run MSP until optimization has effectively converged for every $N$.

Plot the converged errors on log–log axes. For each metric, investigate whether its dependence on $N$ is consistent with

$$\mathcal{E}(N) - \mathcal{E}_\infty \approx cN^{-\beta}. \tag{8}$$

Here, for the recovery error, we can take $\mathcal{E}_\infty=0$ because the setting is noiseless and complete. For the eval loss, estimate $\mathcal{E}_\infty$ using the ground-truth dictionary $\mathbf{A}^\star$: solve (6) with $\mathbf{A}^\star$ and substitute the resulting codes into the eval loss.

Use the smallest two-thirds of the sampled $N$ values for fitting and use the fit to predict the error at the largest $N$. Report the fitted exponent $\beta$ and the extrapolation error. Does the extrapolation error support a power-law description over the measured range?

---

## Part 2: A phase transition in identifiability

Everything so far has kept the codes sparse. Let's now gradually decrease the sparsity level and see what happens. Throughout this part, work in the complete regime using $D = 16$, $\sigma = 0$, $N = 8000$, and at least three seeds for every reported quantity.

### 2.1 The cost of density.

Define $T_\star(\theta)$ as the number of MSP iterations needed to reach $s_k > 0.9$ for at least $95\%$ of atoms. Measure $T_\star$ across $\theta \in \{0.2, 0.4, 0.6, 0.7, 0.8, 0.85\}$ and plot it. Report the spread across seeds and the mean. How does the number of iterations required for convergence change as the codes become denser?

*Note:* This experiment may require a large iteration budget.

### 2.2 The transition.

Keep increasing $\theta$. Sweep $\theta \in \{0.88, 0.90, 0.91, 0.92, 0.93, 0.95, 1.0\}$ and report the fraction of atoms with $s_k > 0.9$. How does this fraction change with $\theta$? Is there evidence of a phase transition?

### 2.3 Can the loss see it?

For each value of $\theta$, report the held-out reconstruction loss of the learned dictionary. Does the reconstruction loss reveal the same transition observed in the previous sub-problem?

### 2.4 Why there?

Now let's try to explain what happened.

a. Compute $\mathbb{E}[z_k^2]$ and $\mathbb{E}[z_k^4]$ under the Part 0 generator, and derive the excess kurtosis $\kappa(\theta) \doteq \mathbb{E}[z_k^4]/(\mathbb{E}[z_k^2])^2 - 3$. Evaluate it at the transition point from 2.2. What is $\kappa(1)$?
b. Show that at $\theta = 1$ the dictionary is unidentifiable.
c. MSP maximizes $\|\mathbf{X}\tilde{\mathbf{A}}^{\top}\|_4^4$ over the orthogonal group. Explain what statistical structure this objective is exploiting, and why that signal vanishes as $\theta \to 1$.

---

## Part 3: Real data dictionary learning

So far, all experiments have used synthetic data, where the ground-truth dictionary $\mathbf{A}$ is known and recovery can be measured directly. Real data provides neither. In this part, investigate which of the previous findings persist on real data. Use either MNIST or CIFAR10 (for CIFAR10, use the first color channel only). Flatten each image into a vector.

### (a) Spectrum and projection.

To reduce the dimensionality of the problem, first compute a low-dimensional projection. Plot the eigenvalue spectrum of $\mathrm{Cov}(\mathbf{x})$ on log axes, and report the number of principal directions required to explain $80\%$, $90\%$, and $99\%$ of the total variance. Choose a reasonable projection dimension $k\le 64$ (recommend $k=32$ for fast computation), project the data onto the top $k$ eigendirections, and precondition the projected data as in Remark 2.8 and Part 0(e). Verify that the preconditioned data has approximately identity covariance before proceeding. From this point on, $k$ plays the role of $D$ in other parts.

### (b) Alternating minimization on the projected data.

Run the alternating minimization of Part 0(c) with both $m=2k$ and $m=4k$. Choose one penalty $\lambda$ for both values of $m$ (recommend $\lambda=2$). Measure the held-out loss (5) on a fixed set of $2000$ test images that is never used for training, use at least three seeds for every reported quantity, and report results for each $m$. Use single precision (`float32`) for efficiency.

**The T-axis.** Fix $N=12{,}000$ and run $400$ blocks, recording the held-out loss and the training objective every $5$ blocks. Plot both quantities on semilogy and log–log axes and characterize their convergence behavior. Compare the held-out loss results with the complete case in Part 1(a) and comment on the similarities or differences.

**The N-axis.** For each seed, fix a random ordering of the training images and, for each $N\in\{200,400,800,1600,3200,6400,12{,}800\}$, use the first $N$ images. For each run, iterate until the relative change in the training objective over the preceding $50$ blocks is below $10^{-5}$, with a cap of $1000$ blocks. Since the asymptotic floor is unknown on real data, treat $\mathcal{E}_\infty$ as a free parameter. Report $\beta$ and the prediction error. Finally, compare $\beta$ found with the sample-size scaling you found on synthetic data in Part 1(b). If the scaling differs, describe how, and explain what may account for the difference.

*Throughout:* compare the held-out loss with that obtained using a random dictionary with the same $k$ and $m$ (rows drawn i.i.d. Gaussian normalized to unit norm), and report their ratio. Since recovery cannot be measured without a ground-truth dictionary, this baseline provides evidence that meaningful learning has occurred. Also report the fraction of nonzero code entries achieved by your chosen penalty.

### (c) Interpret the atoms.

Map the learned atoms back through the inverse preconditioning transformation into pixel space and display them as images for both $m=2k$ and $m=4k$. Are the learned atoms visually interpretable? Does increasing the degree of overcompleteness change their structure or interpretability?

---

## Part 4: Spectral bias in dictionary recovery (Bonus)

*This part is optional and is worth bonus points toward the homework grade.*

In linear regression, gradient descent exhibits spectral learning dynamics: high-variance eigendirections of the input covariance are fitted faster than low-variance ones. In this section, we investigate whether a similar spectral bias arises in the dictionary learning setting.

**The generator for this part.** Work in the complete regime, with $\mathbf{A}\in\mathbb{R}^{m\times D}$, $m=D$, generated as the $\mathbf{Q}$ factor of the QR decomposition of a Gaussian matrix. Replace the Bernoulli–Gaussian model (2) from Part 0 with the following. For each coordinate $k=1,\dots,m$, draw independently

$$z_k = b_k g_k, \qquad b_k \sim \mathrm{Bern}(p_k), \qquad g_k \sim \mathcal{N}(0, v_k), \qquad b_k \text{ and } g_k \text{ independent}, \tag{9}$$

where $p_k\in(0,1]$ is the *activation probability* of atom $k$, and $v_k>0$ is the variance of its coefficient conditional on being active. Set

$$p_k = c\,k^{-\alpha} \quad (c\le 1,\ \alpha \ge 0), \qquad v_k \equiv 1. \tag{10}$$

**Do not rescale $v_k$ across atoms to keep $\mathrm{Var}(z_k)$ constant.** Draw $N$ samples $\mathbf{x}_i = \mathbf{A}^{\top}\mathbf{z}_i + \boldsymbol{\epsilon}_i$ with $\boldsymbol{\epsilon}_i \sim \mathcal{N}(\mathbf{0},\sigma^2\mathbf{I})$ independent of $\mathbf{z}_i$, and write $\mathbf{z}$, $\mathbf{x}$ for generic samples. Define the *per-atom energy*

$$\tau_k \doteq p_k v_k. \tag{11}$$

Before proceeding, report the empirical variance of each code coordinate and verify that it varies across $k$.

Use these settings unless stated otherwise: $c=0.45$, $\alpha=1$, $D=m=16$, $N=12{,}000$, $\sigma=0.01$, LASSO penalty $\lambda = 0.6$, an iteration budget of $T=120$ outer iterations with $L=30$ inner ISTA steps, and at least three random seeds for every reported quantity. Recall from Part 0 that $s_k \doteq \max_j |\langle \mathbf{a}_k, \hat{\mathbf{a}}_j\rangle|$ measures the recovery of atom $k$. Define the *recovery time* $t_k$ as the first outer iteration for which $s_k>0.8$, and leave $t_k$ undefined if this threshold is never reached.

### 4.1 (Derivation).

a. For fixed $c$, how does the expected sparsity level change as $\alpha$ increases?
b. Show that under (9), $\mathbb{E}[\mathbf{z}]=\mathbf{0}$ and $\mathbb{E}[\mathbf{z}\mathbf{z}^\top] = \mathrm{diag}(\tau_1,\dots,\tau_m)$.
c. Deduce that $\mathrm{Cov}(\mathbf{x}) = \mathbf{A}^\top\mathrm{diag}(\tau_1,\dots,\tau_m)\mathbf{A} + \sigma^2\mathbf{I}$.
d. Show that, in the complete orthogonal regime, each atom $\mathbf{a}_k$ is an eigenvector of $\mathrm{Cov}(\mathbf{x})$, and compute the corresponding eigenvalue.
e. Show that if instead $v_k=1/p_k$, then $\tau_k\equiv1$ for all $k$ and all $\alpha$. What does this imply about using the Part 0 generator to study spectral bias?

### 4.2 Ordering.

Using the alternating minimization method, plot $t_k$ against $\tau_k$ on log–log axes for $\alpha \in \{0.75,\,1.0,\,1.25\}$ at fixed $c=0.45$. Does a clear ordering appear, with higher-energy atoms recovered earlier, and does it strengthen with $\alpha$? Report the variation across seeds and assess whether it is small or large relative to the observed trend. Show atoms that never reach the threshold within $T$ iterations separately, e.g. as markers above the budget line.

### 4.3 Does the algorithm matter?

Replace alternating minimization with the MSP algorithm and, using identical data, produce the same plot as in 4.2 at $\alpha=1$, shown side by side with the corresponding alternating minimization plot. Report the range of recovery times for both methods. Does the ordering persist? Explain what property of MSP accounts for the behavior you observe.

---

## Writeup

Congratulations! You have completed Assignment 1. Prepare a writeup in LaTeX, compile it to PDF, and include:

- The URL of your private repository in the `hku-data8014-students` GitHub organization;
- A brief description of your code change to accommodate this homework;
- The required plots and accompanying explanations for each part of the assignment;
- A disclosure of your AI usage. Using AI is permitted; please explain where and how you used it.

Push your final code to GitHub. Then open Assignment 1 on <https://data8014.com>, upload your PDF writeup, and enter your repository name and the full commit SHA of the version you want graded. You can obtain the SHA by running `git rev-parse HEAD` in your local repository.

Click **Submit current work** and verify that a submission receipt appears. **Uploading files or sending chat messages alone does not submit your assignment. Your submission preserves the specified code version and writeup**; subsequent GitHub pushes will not change it.

Submit by the deadline displayed on the assignment page.

⏰ **Deadline:** Oct 1, 2026, 11:59 PM HKT
