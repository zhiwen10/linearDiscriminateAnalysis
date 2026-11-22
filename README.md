# Linear Discriminant Analysis

A visual, step-by-step introduction to linear discriminant analysis (LDA), using simulated two-class data and the Iris dataset for a multiclass example.

The tutorial covers class means and variance, within-class scatter, geometric projection, Fisher's criterion, covariance adjustment, whitening, decision boundaries, Lagrangians, and generalized eigenvalues.

## Repository contents

- `LDA_Tutorial.ipynb` — the single tutorial notebook with saved outputs.
- `lda_plotting.py` — plotting helpers imported by the notebook so its mathematical code stays easy to read.
- `pyproject.toml` — Python requirements and package-index configuration.
- `uv.lock` — locked dependency versions for reproducible installation.

## Requirements

- Python 3.11 or newer.
- [`uv`](https://docs.astral.sh/uv/) for environment and package management.

Check that `uv` is available:

```bash
uv --version
```

## Installation

Clone the repository and create the environment:

```bash
git clone https://github.com/zhiwen10/linearDiscriminateAnalysis.git
cd linearDiscriminateAnalysis
UV_HTTP_TIMEOUT=120 uv sync
```

The `uv sync` command creates `.venv` and installs NumPy, SciPy, Matplotlib, scikit-learn, and Jupyter.

### Python package mirrors in China

This project is already configured to use Tsinghua's PyPI mirror. No global pip or `uv` configuration is changed. The setting is stored in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "tsinghua"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
```

If Tsinghua is unavailable, try another China mirror without editing the project:

```bash
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ UV_HTTP_TIMEOUT=120 uv sync
```

or:

```bash
UV_DEFAULT_INDEX=https://pypi.mirrors.ustc.edu.cn/simple/ UV_HTTP_TIMEOUT=120 uv sync
```

To use the official PyPI index instead:

```bash
UV_DEFAULT_INDEX=https://pypi.org/simple uv sync
```

## Run the tutorial

Start JupyterLab with the notebook:

```bash
uv run jupyter lab LDA_Tutorial.ipynb
```

Use the **Python 3** kernel and choose **Run All**. The examples use a fixed random seed, and Iris is bundled with scikit-learn, so the notebook does not download any datasets.

To execute the notebook from the command line instead:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace LDA_Tutorial.ipynb
```
