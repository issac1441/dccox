# dccox

[![CI](https://github.com/issac1441/dccox/actions/workflows/ci.yml/badge.svg)](https://github.com/issac1441/dccox/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-ARL--1.1-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

The PoC of DC-COX, a federated Cox PH regression approach.


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


## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.


## 📄 License

This project is licensed under the ARL-1.1. See the [LICENSE](LICENSE) file for details.
