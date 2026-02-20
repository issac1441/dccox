# DC-COX

[![CI](https://github.com/issac1441/dccox/actions/workflows/ci.yml/badge.svg)](https://github.com/issac1441/dccox/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-ARL--1.1-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

An **UNOFFICIAL** implementation of [DC-COX](https://www.sciencedirect.com/science/article/pii/S1532046422002696), a federated Cox PH regression approach.

> [!IMPORTANT]
> This project is **not affiliated with or endorsed by** the original authors or the publisher. The DC-COX method is credited to the original paper; this repository provides an independent software implementation.

## Differences from the DC-COX paper

This repository implements the DC-COX workflow and follows the paper’s core algorithms, but includes several **engineering/stability** choices and **output conventions** that can differ from a strict paper-only implementation.

### 1) Bootstrap sampling default: subsampling without replacement (`bs_replace=False`)
- The paper’s Algorithm 2 describes bootstrap-based dimensionality reduction.
- This implementation defaults to **subsampling without replacement** (`bs_replace=False`) to improve numerical stability when fitting Cox models with `lifelines` (fewer duplicated rows → lower risk of collinearity / singularity convergence failures).
- You can switch to classic bootstrap by setting `bs_replace=True`.

### 2) PCA-based $F_{DR}$: feature-space basis via SVD
- The paper allows combining bootstrap-based DR with other DR methods (e.g., PCA/LPP/NMF) to form:
  
  <div align="center">

  $$F = [F_{BS}, F_{DR}]E.$$
  $$F_{DR} \in \mathbb{R}^{m\times \tilde m_{DR}}$$

  </div>

- This implementation constructs $F_{DR} \in \mathbb{R}^{m\times \tilde m_{DR}}$ using a **feature-space PCA basis** computed via SVD (i.e., PCA loadings / directions).

### 3) Random nonsingular matrix $E$: orthogonal matrix via QR
- The paper requires $E$ to be a **random nonsingular matrix**.
- This implementation uses a **random orthogonal matrix** (via QR decomposition) as a numerically stable special case of nonsingular matrices.

### 4) Master-side stability: dropping low-variance columns in $\hat X$
- The paper fits Cox on $\hat X$ directly.
- This implementation optionally drops columns of $\hat X$ with variance below `var_thres` (default `1e-8`) before fitting, to reduce degeneracy/singularity issues in `lifelines`.
- When columns are dropped, the corresponding columns in $G$ are dropped as well to keep mappings consistent.

  > [!NOTE]
  > This can cause differences from a strict centralized reference if the centralized run would keep those columns.

### 5) Survival prediction supports two centering conventions (`centering`)
- The paper uses the standard Cox form:

  <div align="center">
  
  $$h(t\mid x) = h_0(t)\exp(x^\top \beta)$$

  </div>

  which corresponds to defining baseline hazard at $x=0$.

- `lifelines` reports baseline hazard under a mean-centered convention. To support both, this implementation provides:

  **(a) `centering=None` (paper convention)**  
  Uses $x^\top\beta$ in the linear predictor, and rescales the provided baseline hazard by a constant factor:
  
  <div align="center">

  $$h_0^{paper}(t) = h_0^{lifelines}(t)\exp(-\bar x^\top\beta)$$
  
  </div>

  Partial hazard uses $\exp(x^\top\beta)$.

  **(b) `centering="mean"` (lifelines convention)**  
  Uses $(x-\bar x)^\top\beta$ in the linear predictor and keeps baseline hazard unchanged:
  
  <div align="center">
  
  $$h(t\mid x) = h_0^{lifelines}(t)\exp((x-\bar x)^\top\beta)$$
  
  </div>

- Both conventions yield **identical final hazard/survival predictions**, but the *reported* baseline hazard differs by a constant factor.

### 6) Extra artifact: returning `feature_mean`
- The paper’s core artifacts are coefficients (and variance) plus baseline hazard.
- This implementation additionally computes/returns the **global feature mean** $\bar x$ (`feature_mean`) for prediction/reporting convenience (used by the `centering` option). This is not a core artifact explicitly described in the paper.

### 7) Engineering-oriented I/O and data structures
- The paper describes conceptual steps across parties.
- This repo provides practical wrappers and data structures (e.g., `BlockMatrix`, `usecase.Horizontal`) for orchestration, missing-value handling, and commercial/workflow integration.


## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/issac1441/dccox
cd dccox

# Install dependencies using uv
uv sync --all-groups
```


## 🚀 Quick Start

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the application
python -m dccox.main
```


## 🛠️ Development

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager

### Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install all dependencies** (including dev, test, docs):
   ```bash
   uv sync --all-groups
   ```

3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### 🧪 Testing

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=dccox
```

### 🧹 Linting & Formatting

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

```bash
# Check for linting errors
uv run ruff check .

# Auto-fix linting errors
uv run ruff check . --fix

# Format code
uv run ruff format .
```

### 📝 Type Checking (Optional)

```bash
uv run mypy dccox
```

### 📚 Documentation

Build documentation using Sphinx:

```bash
cd docs
make html
```

The generated documentation will be in `docs/_build/html/`.


### 🐳 Docker

```bash
# Build
docker build -t dccox .

# Run
docker run dccox
```


## 📁 Project Structure

```
dccox/
├── dccox/          # Main package
│   ├── __init__.py
│   └── main.py
├── tests/                   # Test files
├── docs/                    # Sphinx documentation
├── .github/                 # GitHub Actions & templates
│   ├── workflows/
│   └── ISSUE_TEMPLATE/
├── pyproject.toml           # Project configuration
├── Dockerfile
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
└── CHANGELOG.md
```

## 📚 References & Citation

### Original Paper (Method)
If you use the DC-COX method in your research, please cite the original paper:
> [DC-COX: Data collaboration Cox proportional hazards model for privacy-preserving survival analysis on multiple parties](https://www.sciencedirect.com/science/article/pii/S1532046422002696)

```bibtex
@article{imakura2023dccox,
  title={DC-COX: Data collaboration Cox proportional hazards model for privacy-preserving survival analysis on multiple parties},
  author={Imakura, Akira and Ye, Xiucai and Ma, Xinyu and Arai, Watchara and Sakurai, Tetsuya},
  journal={Journal of Biomedical Informatics},
  volume={137},
  pages={104264},
  year={2023},
  doi={10.1016/j.jbi.2022.104264}
}
```

### This Implementation (Software)
If you use this specific software implementation, please cite it using the metadata in [`CITATION.cff`](CITATION.cff) or as follows:
> Wen, J.-H. (2026). dccox: An unofficial implementation of DC-COX [Computer software]. https://github.com/issac1441/dccox

```bibtex
@software{wen2026dccox,
  author = {Wen, Jian-Hung},
  title = {dccox: An unofficial implementation of DC-COX},
  url = {https://github.com/issac1441/dccox},
  year = {2026}
}
```


## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.


## 📄 License

This project is licensed under the ARL-1.1. See the [LICENSE](LICENSE) file for details.

> **Note on Third-party Components**: This software depends on third-party open-source packages (including `lifelines`, `numpy`, `pandas`, `pydantic`, `scipy`) distributed under their respective licenses. Please refer to `NOTICE` for details.
