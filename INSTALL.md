# Installation Guide

## Quick Start

### Option 1: Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/adhal007/OmixHub.git
cd OmixHub

# Create and activate environment
conda env create -f environment.yml
conda activate omixhub

# Verify installation
python -c "import torch; import scanpy; import pandas; print('Success!')"
```

### Option 2: Minimal Install

For a lighter installation without optional dependencies:

```bash
conda env create -f environment-minimal.yml
conda activate omixhub-minimal
```

### Option 3: Pip Only

If conda is not available:

```bash
# Create virtual environment
python -m venv omixhub-env
source omixhub-env/bin/activate  # On Windows: omixhub-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Platform-Specific Notes

### Apple Silicon (M1/M2/M3)

PyTorch will automatically use the Metal Performance Shaders (MPS) backend:

```python
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
```

### NVIDIA GPU (CUDA)

For GPU acceleration, install PyTorch with CUDA:

```bash
# After creating the conda environment
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

Verify CUDA:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

### Windows

Use Anaconda Prompt or PowerShell with conda initialized:

```powershell
conda env create -f environment.yml
conda activate omixhub
```

---

## Optional Dependencies

### CellxGene Census (Single-cell data)

Already included in full environment. For manual install:

```bash
pip install cellxgene-census
```

Test:
```python
import cellxgene_census
print("CellxGene Census installed!")
```

### Google BigQuery

For accessing BigQuery datasets:

```bash
pip install google-cloud-bigquery

# Authenticate
gcloud auth application-default login
```

### MongoDB

For local database storage:

```bash
pip install pymongo

# Requires MongoDB server running locally or connection string
```

---

## Troubleshooting

### Common Issues

**1. `conda: command not found`**

Install Miniconda:
```bash
# macOS/Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-$(uname -s)-$(uname -m).sh
bash Miniconda3-latest-*.sh
```

**2. Environment creation fails with conflicts**

Try the minimal environment:
```bash
conda env create -f environment-minimal.yml
```

Or create a fresh environment and install manually:
```bash
conda create -n omixhub python=3.11
conda activate omixhub
pip install -r requirements.txt
```

**3. PyTorch not finding GPU**

Check CUDA version compatibility:
```bash
nvidia-smi  # Shows CUDA version
```

Install matching PyTorch:
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**4. `scanpy` import errors**

May need additional dependencies:
```bash
pip install igraph leidenalg
```

**5. `hdbscan` installation fails**

Install build dependencies first:
```bash
conda install -c conda-forge hdbscan
```

---

## Verifying Installation

Run this script to check all components:

```python
#!/usr/bin/env python
"""Verify OmixHub installation."""

def check_imports():
    modules = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'torch': 'pytorch',
        'scanpy': 'scanpy',
        'anndata': 'anndata',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'tqdm': 'tqdm',
    }
    
    print("Checking required packages...")
    all_ok = True
    
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            all_ok = False
    
    # Optional packages
    print("\nChecking optional packages...")
    optional = ['cellxgene_census', 'pydeseq2', 'shap', 'streamlit']
    
    for module in optional:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ○ {module} - not installed (optional)")
    
    # Check PyTorch device
    print("\nPyTorch device check...")
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("  ✓ MPS (Apple Silicon) available")
    else:
        print("  ○ CPU only (GPU not detected)")
    
    return all_ok

if __name__ == "__main__":
    check_imports()
```

Save as `check_install.py` and run:
```bash
python check_install.py
```

---

## Updating

To update an existing environment:

```bash
conda activate omixhub
conda env update -f environment.yml --prune
```

Or for pip:
```bash
pip install -r requirements.txt --upgrade
```
