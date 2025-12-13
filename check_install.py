#!/usr/bin/env python
"""
OmixHub Installation Verification
=================================

Run this script to verify your installation:
    python check_install.py

Exit codes:
    0 - All required packages installed
    1 - Missing required packages
"""

import sys


def check_package(module_name: str, package_name: str = None) -> bool:
    """Check if a package is importable."""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def get_version(module_name: str) -> str:
    """Get package version."""
    try:
        module = __import__(module_name)
        return getattr(module, '__version__', 'unknown')
    except:
        return 'N/A'


def main():
    print("=" * 60)
    print("OmixHub Installation Check")
    print("=" * 60)
    
    # Required packages
    required = {
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
        'requests': 'requests',
        'h5py': 'h5py',
    }
    
    # Optional packages
    optional = {
        'cellxgene_census': 'cellxgene-census (single-cell data)',
        'pydeseq2': 'pydeseq2 (differential expression)',
        'shap': 'shap (model interpretation)',
        'umap': 'umap-learn (dimensionality reduction)',
        'hdbscan': 'hdbscan (clustering)',
        'streamlit': 'streamlit (web app)',
        'pymongo': 'pymongo (MongoDB)',
        'google.cloud.bigquery': 'google-cloud-bigquery',
    }
    
    # Check required
    print("\n📦 Required Packages:")
    print("-" * 40)
    
    missing_required = []
    for module, name in required.items():
        if check_package(module):
            version = get_version(module)
            print(f"  ✓ {name:<20} {version}")
        else:
            print(f"  ✗ {name:<20} NOT INSTALLED")
            missing_required.append(name)
    
    # Check optional
    print("\n📦 Optional Packages:")
    print("-" * 40)
    
    for module, name in optional.items():
        if check_package(module):
            print(f"  ✓ {name}")
        else:
            print(f"  ○ {name} (not installed)")
    
    # Check compute device
    print("\n🖥️  Compute Device:")
    print("-" * 40)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"  ✓ CUDA GPU: {device_name}")
            print(f"    CUDA version: {cuda_version}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  ✓ Apple Silicon (MPS) available")
        else:
            print("  ○ CPU only (no GPU acceleration)")
        
        print(f"    PyTorch version: {torch.__version__}")
    except ImportError:
        print("  ✗ PyTorch not installed")
    
    # Check GDC connection
    print("\n🌐 API Connectivity:")
    print("-" * 40)
    
    try:
        import requests
        response = requests.get("https://api.gdc.cancer.gov/status", timeout=5)
        if response.status_code == 200:
            print("  ✓ GDC API reachable")
        else:
            print("  ○ GDC API returned status:", response.status_code)
    except Exception as e:
        print(f"  ○ GDC API check failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if missing_required:
        print("❌ MISSING REQUIRED PACKAGES:")
        for pkg in missing_required:
            print(f"   - {pkg}")
        print("\nInstall with:")
        print("  conda env update -f environment.yml")
        print("  # or")
        print("  pip install -r requirements.txt")
        return 1
    else:
        print("✅ All required packages installed!")
        print("\nYou're ready to use OmixHub.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
