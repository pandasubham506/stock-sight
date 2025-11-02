# Stock Market Prediction - Installation Guide

## Python Version

**Use Python 3.8.10** (3.8.x series)

This project has strict dependency constraints:
- TensorFlow 2.3.4 requires NumPy < 1.19.0
- Newer packages require Python >= 3.8
- **Result**: Python 3.8.10 is the optimal version

## Installation

### 1. Create and activate virtual environment

```powershell
# Create .venv with Python 3.8
py -3.8 -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
# Upgrade pip
python -m pip install -U pip setuptools wheel

# Install from requirements
pip install -r REQUIREMENTS.txt
```

## What Changed from Original Requirements

| Package | Original | Fixed | Reason |
|---------|----------|-------|--------|
| numpy | 1.22.0 | 1.18.5 | TensorFlow 2.3.x requires <1.19.0 |
| pandas | 1.1.0 | 1.1.5 | Bug fix release, still compatible |
| scipy | 1.10.0 | 1.5.4 | Compatible with numpy 1.18.x |
| matplotlib | 3.3.1 | 3.3.4 | Bug fix release |
| plotly | 4.9.0 | 4.14.3 | Bug fix release |
| yfinance | 0.1.54 | 0.1.70 | Bug fixes, better data fetching |
| tensorboard | 2.3.0 | (removed) | Let TensorFlow manage it |
| fbprophet | 0.6 | (removed) | Build fails on Windows with this numpy |

## About fbprophet / Prophet

The notebooks reference `fbprophet` for time series forecasting. The original `fbprophet==0.6` **cannot be installed** in this environment due to:

1. Windows build issues (requires C++ compiler and Stan toolchain)
2. Its dependency `cmdstanpy` requires `numpy>=1.21`, conflicting with TensorFlow 2.3.x's `numpy<1.19`

### Workaround Options

**Option 1: Skip Prophet**
- Most of the project works without it
- Only affects forecast notebooks in `src/notebooks/`

**Option 2: Separate Prophet Environment** (Recommended if you need it)
```powershell
# Create a second environment just for Prophet notebooks
py -3.10 -m venv .venv-prophet
.\.venv-prophet\Scripts\Activate.ps1
pip install prophet pandas matplotlib numpy jupyter
```

**Option 3: Use Pre-built Prophet Wheel**
- Download a compatible Windows wheel from unofficial builds
- Not recommended for production

## Verification

Test your installation:

```powershell
python -c "import tensorflow; import numpy; import pandas; import flask; print('All core imports successful')"
```

Expected output:
```
All core imports successful
```

## Running the Application

```powershell
# Make sure you're in the .venv environment
python runserver.py
```

---
*Fixed for Python 3.8.10 on Windows - November 2025*
