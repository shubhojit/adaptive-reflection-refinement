# Spatially Adaptive Reflection Refinement

A small CPU-only teaching project about allocating a limited image-refinement budget.
It asks:

> Given the same initial reflection-removal estimate and the same number of region
> updates, does choosing regions adaptively improve reconstruction quality more than
> spreading those updates uniformly across the image?

This is an educational prototype, not a state-of-the-art reflection remover. It uses
synthetic scenes so that the clean transmission layer is known, a deliberately simple
initial estimate, and a hand-written NumPy refinement objective. It does not require
a GPU, pretrained weights, or a dataset download.

## What the experiment teaches

- A single observed image does not uniquely determine its transmission and reflection.
- Gradient exclusion is a prior, not a physical law; it can remove real scene detail.
- A region router can help only when its score identifies places where refinement helps.
- Equal region-update counts do not imply equal wall-clock time: routing has overhead.
- Synthetic results are controlled teaching evidence, not proof of real-world performance.

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── src/
│   └── reflection_refinement/
│       ├── __init__.py
│       ├── baseline.py
│       ├── experiment.py
│       ├── metrics.py
│       ├── refinement.py
│       ├── routing.py
│       └── synthetic.py
├── experiments/
│   ├── 01_sanity_check.py
│   ├── 02_budget_sweep.py
│   └── 03_reflection_strength_sweep.py
├── tests/
│   └── test_core.py
└── results/
    └── .gitkeep
```

## Setup (CPU only)

With the Conda environment created for this project:

```bash
conda activate reflection-removal
python -m pip install -e .
```

Alternatively, use any Python 3.10+ environment and install the local project with
`python -m pip install -e .`. NumPy is the only runtime dependency.

## Run

Run a compact sanity check:

```bash
python experiments/01_sanity_check.py
```

Compare adaptive routing with uniform spatial allocation over a range of budgets:

```bash
python experiments/02_budget_sweep.py --images 20 --size 128 --output results/budget_sweep.csv
```

Vary reflection strength while using the same generated scenes at each strength:

```bash
python experiments/03_reflection_strength_sweep.py --images 10 --size 128 --strengths 0.1 0.28 0.45 0.65
```

For a parameter experiment:

```bash
python experiments/02_budget_sweep.py --images 20 --size 128 --learning-rate 0.2 --lambda-exclusion 100000
```

The budget sweep writes one CSV row per image, budget, and routing policy. The strength
sweep adds a controlled reflection-strength variable. Both report initial and refined
MAE/PSNR, selected tile count, the nominal update budget, and measured wall-clock time.

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Model and objective

For a clean transmission image \(T\), reflection image \(R\), and observed mixture \(I\),
the generator uses the simple approximation

\[
I = (1-\alpha)T + \alpha R,
\]

with clipping to the display range. The clipping means the equality may not hold after
clipping at saturated pixels.

The toy initial estimator subtracts a scaled, blurred version of the observation:

\[
T_0 = \operatorname{clip}\bigl(I-\beta\,G_\sigma(I),\,0,\,1\bigr).
\]

This is intentionally crude. It gives the class a common starting point without a
pretrained network. It is not intended to model a competitive reflection-removal
algorithm.

For a candidate transmission \(T\), the residual is \(R(T)=I-T\). With forward finite
differences \(D_x,D_y\), define the per-pixel gradient inner product

\[
q_p(T) =
\sum_c \left[
(D_xT)_{p,c}(D_x(I-T))_{p,c}
+ (D_yT)_{p,c}(D_y(I-T))_{p,c}
\right].
\]

The refinement objective is

\[
\mathcal{L}(T)=
\lambda_{\mathrm{fid}}\frac{\lVert T-T_0\rVert_2^2}{HWC}
+ \lambda_{\mathrm{excl}}\frac{1}{HW}\sum_p q_p(T)^2.
\]

The first term keeps refinement near the initial estimate. The second penalizes
coincident, similarly oriented transmission/residual gradients. This is a deliberately
small illustrative objective: a small loss does **not** imply a correct physical
decomposition.

The default exclusion weight is `100000` because finite-difference gradients and their
products are small for images in `[0, 1]`. It is only a starting value; students should
sweep it alongside the learning rate and inspect both metrics and visual results.

## Routing and budget

The router scores each fixed-size tile by mean absolute gradient overlap in \(T_0\) and
\(I-T_0\). Scores come only from the observation and initial estimate; clean targets
are used only for evaluation.

- **Adaptive:** refine the tiles with the largest score.
- **Uniform:** choose tiles evenly across the image, independent of their score.

Each selected tile receives one gradient-descent update. The nominal budget is
`selected_tiles * tile_height * tile_width` updated pixels per image, equal for the two
policies. A one-pixel halo supplies neighboring pixels for finite differences; each
tile's halo is reflect-padded at the image boundary, so every selected tile has the
same refinement crop size. The CSV reports both updated pixels and processed crop
pixels. Routing overhead is not included in the refinement budget, so the CSV also
records end-to-end elapsed time; adaptive routing pays for computing its scores.

This budget is a refinement-work proxy, not a hardware-independent FLOP measure. NumPy
may compute small patches less efficiently than large arrays; results should include
timing and avoid claiming speedup from the nominal budget alone.
The sweep requires image size to be divisible by tile size, so every selected tile has
the same area and the two policies receive the same updated-pixel count.

## Important limitations

- The synthetic generator uses unrelated simple patterns, not realistic glass optics.
- The initial estimate is intentionally weak and may bias which regions look useful.
- A single update per selected tile favors the first experiment's simplicity over
  convergence.
- Independent local updates can behave differently near tile boundaries; the halo
  reduces derivative-boundary artifacts but does not eliminate every tiling effect.
- MAE and PSNR against synthetic ground truth do not fully capture perceptual quality.
- Adaptive routing may lose to uniform allocation. The project does not assume a win.

## Suggested classroom extensions

1. Change reflection strength, blur, and displacement, then group results by condition.
2. Change the tile score and compare against random routing.
3. Sweep learning rate and the two loss weights; plot quality against budget.
4. Add a full-image refinement baseline and account for its actual work.
5. Compare RGB-gradient exclusion with luminance-gradient exclusion.
6. Try FFT-based analysis as a separate experiment; FFT is not required by the core
   spatial-routing hypothesis.

## Reference context

Spatially adaptive computation and adaptive depth already have prior work, including
Figurnov et al., [Spatially Adaptive Computation Time for Residual Networks (CVPR
2017)](https://openaccess.thecvf.com/content_cvpr_2017/html/Figurnov_Spatially_Adaptive_Computation_CVPR_2017_paper.html),
and Liu et al., [Deep Adaptive Inference Networks for Single Image Super-Resolution
(2020)](https://arxiv.org/abs/2004.03915). This project is an instructional experiment
about applying equal-budget spatial refinement to reflection removal, not a novelty
claim about adaptive computation in general.
