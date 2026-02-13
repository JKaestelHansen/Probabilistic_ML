# CLAUDE.md

## Project Overview

This is a research repository for **probabilistic machine learning** applied to predicting diffusional properties — specifically **Cohesin extrusion speeds** from DNA loci pair movement trajectories. The project combines molecular dynamics simulations with deep learning and comprehensive uncertainty quantification.

**Key reference:** [Kæstel-Hansen et al., PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012061#sec020)

## Repository Structure

```
Probabilistic_ML/
├── CLAUDE.md                                # This file
├── README.md                                # Project documentation and metrics overview
├── DL_MCDropout_predictions.py              # Main DL script: CNN-MLP with MC Dropout
├── DL_multiSWAG_predictions.py              # Alternative DL script: SWAG uncertainty method
├── helper_functions.py                      # Feature engineering utilities
├── simulate_brownian_directed_mixture.py    # Brownian motion + directed motion simulation
├── 230610_Gen1D.py                          # 1D trajectory generation (polychrom)
├── 230706_Sim3D.py                          # 3D molecular dynamics simulation (OpenMM)
├── Analyze_Sim_3D.ipynb                     # Jupyter notebook for 3D simulation analysis
└── uncertainty_quantification/              # Reusable UQ metrics package
    ├── __init__.py
    ├── calibration.py                       # Calibration metrics (ECE, MCE, ENCE, reliability)
    ├── confidence.py                        # Confidence curves (AUCO, error drop, ranking)
    ├── chi_squared.py                       # Chi-squared statistics (reduced, ANEES)
    └── README                               # Package reference link
```

## Tech Stack and Dependencies

**Language:** Python 3

**Core dependencies** (no requirements.txt — imports are inline):
- `torch` — Neural network training and inference
- `numpy`, `scipy`, `pandas` — Numerical and scientific computing
- `matplotlib`, `plotly` — Visualization (2D and 3D)
- `scikit-learn` — Train/test splitting, preprocessing
- `tqdm` — Progress bars
- `h5py` — HDF5 data I/O

**Simulation-specific** (only needed for `230610_Gen1D.py` and `230706_Sim3D.py`):
- `polychrom` — Polymer/chromatin simulation toolkit
- `openmm` / `simtk` — Molecular dynamics engine

## Key Architecture

### Neural Network (CNN-MLP)

Defined in `DL_MCDropout_predictions.py` and `DL_multiSWAG_predictions.py`:

- **Class:** `CNN_MLP(nn.Module)` — Conv1d layers → ReLU → MaxPool → fully connected layers
- **Input:** Variable-length trajectory sequences (padded to max length)
- **Output:** Mean prediction + learned variance (aleatoric uncertainty)
- **Uncertainty:** MC Dropout (multiple stochastic forward passes at test time) or SWAG

### Dataset Handling

- **Class:** `Tracks_to_Dataset(Dataset)` — Pads variable-length sequences, converts to tensors
- **Data format:** Pickle files at `data/directed_tracks/tracks.pkl` and `speeds.pkl`
- Training data is not included in the repository

### Feature Engineering (`helper_functions.py`)

Pipeline via `add_features(X, features_list)` supporting:
- `squaredist_full` — Euclidean distances
- `dotproduct_traces` — Vector dot products
- `steplength_traces` — Step lengths (2D/3D)
- `origin_distance` — Distance from starting point
- `euclidian_coordinates_to_polar_or_sphere` — Coordinate transforms

### Uncertainty Quantification Package (`uncertainty_quantification/`)

Modular, reusable package exporting:

**Calibration** (`calibration.py`):
- `confidence_based_calibration()` — Confidence interval calibration
- `error_based_calibration()` — Binned error vs uncertainty
- `expected_calibration_error()` — ECE
- `max_calibration_error()` — MCE
- `expected_normalized_calibration_error()` — ENCE
- `prep_reliability_diagram()` — Reliability diagram data

**Confidence** (`confidence.py`):
- `ranking_confidence_curve()` — Confidence curves by filtering high-uncertainty predictions
- `area_confidence_oracle_error()` — AUCO metric
- `error_drop()` — Range between best/worst uncertainty quantiles
- `decreasing_ratio()` — Monotonicity check

**Chi-squared** (`chi_squared.py`):
- `chi_squared_stat()`, `reduced_chi_squared_stat()`, `chi_squared_anees()`

## Code Conventions

- **Naming:** `snake_case` for functions/variables, `CamelCase` for classes (e.g., `CNN_MLP`, `Tracks_to_Dataset`)
- **Script format:** Main DL scripts use Jupyter `# %%` cell separators (runnable as notebooks in VS Code or Jupyter)
- **Device handling:** Scripts detect MPS (Apple Silicon) with CPU fallback; CUDA supported implicitly via PyTorch
- **No type hints** in existing code
- **No tests or CI/CD** — this is a research codebase

## Data Flow

```
Simulation (polychrom/OpenMM) → HDF5/Pickle files
    → Feature engineering (helper_functions.py)
    → Train/Val/Test split (sklearn)
    → PyTorch DataLoader (Tracks_to_Dataset)
    → CNN-MLP training with uncertainty-aware loss
    → MC Dropout / SWAG inference
    → Uncertainty calibration evaluation (uncertainty_quantification/)
```

## Common Tasks

### Running the main deep learning pipeline

The DL scripts are designed to be run cell-by-cell (Jupyter-style) or as full scripts:

```bash
python DL_MCDropout_predictions.py
python DL_multiSWAG_predictions.py
```

These expect training data in `data/directed_tracks/`.

### Running simulations

```bash
python 230610_Gen1D.py      # 1D trajectory generation
python 230706_Sim3D.py      # 3D molecular dynamics
python simulate_brownian_directed_mixture.py  # Brownian motion
```

Simulation scripts require `polychrom` and `openmm`.

### Using the uncertainty_quantification package

```python
from uncertainty_quantification import (
    confidence_based_calibration,
    ranking_confidence_curve,
    area_confidence_oracle_error,
    error_drop,
)
```

## Important Notes for AI Assistants

- **Research code:** Prioritize clarity and correctness over production patterns. Do not add unnecessary abstraction.
- **No package manager config:** Dependencies are not formally declared. If adding new imports, note them clearly in commit messages.
- **External dependencies not in repo:** `SMC_class` (imported in `230610_Gen1D.py`) and training data files are not included.
- **Notebook-style scripts:** The `# %%` markers in `.py` files are intentional — they enable cell-by-cell execution in VS Code / Jupyter.
- **UQ package is reusable:** The `uncertainty_quantification/` package is designed to be portable. Keep it self-contained with minimal dependencies (numpy, scipy).
- **Loss functions:** The project uses learned aleatoric uncertainty via Gaussian NLL loss combined with standard MSE. Modifications to loss functions should preserve the dual epistemic+aleatoric uncertainty framework.
