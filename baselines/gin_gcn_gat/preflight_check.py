"""
Pre-Flight Checklist for Phase 1 GNN Baseline Experiments

Run this before starting experiments to verify:
1. Environment is properly configured
2. Data files exist and are formatted correctly
3. Output directories are writable
4. GPU is accessible (if available)

Author: Sapta (林恩)
Date: 2026-04-01
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path

# ANSI color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'


def check_mark(passed):
    """Return colored check mark or X"""
    return f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"


def print_header(text):
    """Print formatted section header"""
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}{text}{RESET}")
    print(f"{BOLD}{'='*80}{RESET}\n")


def check_environment():
    """Check Python environment and packages"""
    print_header("1. ENVIRONMENT CHECK")
    
    checks = []
    
    # Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info.major == 3 and sys.version_info.minor >= 8
    checks.append(('Python version', py_version, py_ok, 'Requires Python 3.8+'))
    
    # PyTorch
    try:
        import torch
        torch_version = torch.__version__
        torch_ok = True
    except ImportError:
        torch_version = "Not installed"
        torch_ok = False
    checks.append(('PyTorch', torch_version, torch_ok, 'Required'))
    
    # PyTorch Geometric
    try:
        import torch_geometric
        pyg_version = torch_geometric.__version__
        pyg_ok = True
    except ImportError:
        pyg_version = "Not installed"
        pyg_ok = False
    checks.append(('PyTorch Geometric', pyg_version, pyg_ok, 'Required'))
    
    # RDKit
    try:
        from rdkit import Chem
        from rdkit import __version__ as rdkit_version
        rdkit_ok = True
    except ImportError:
        rdkit_version = "Not installed"
        rdkit_ok = False
    checks.append(('RDKit', rdkit_version, rdkit_ok, 'Required'))
    
    # NumPy
    try:
        import numpy as np
        numpy_version = np.__version__
        numpy_ok = True
    except ImportError:
        numpy_version = "Not installed"
        numpy_ok = False
    checks.append(('NumPy', numpy_version, numpy_ok, 'Required'))
    
    # Pandas
    try:
        import pandas as pd
        pandas_version = pd.__version__
        pandas_ok = True
    except ImportError:
        pandas_version = "Not installed"
        pandas_ok = False
    checks.append(('Pandas', pandas_version, pandas_ok, 'Required'))
    
    # SciPy (for statistical tests)
    try:
        import scipy
        scipy_version = scipy.__version__
        scipy_ok = True
    except ImportError:
        scipy_version = "Not installed"
        scipy_ok = False
    checks.append(('SciPy', scipy_version, scipy_ok, 'Required for stats'))
    
    # Print results
    for name, version, ok, note in checks:
        status = check_mark(ok)
        print(f"{status} {name:<25} {version:<20} {note}")
    
    all_ok = all(check[2] for check in checks)
    
    if not all_ok:
        print(f"\n{RED}Environment check FAILED. Install missing packages.{RESET}")
        print(f"\nQuick fix:")
        print(f"  conda activate molprop")
        print(f"  pip install torch torch_geometric rdkit scipy --break-system-packages")
    
    return all_ok


def check_gpu():
    """Check GPU availability"""
    print_header("2. GPU CHECK")
    
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"{check_mark(True)} CUDA available: {torch.version.cuda}")
        print(f"{check_mark(True)} GPU count: {gpu_count}")
        print(f"{check_mark(True)} GPU name: {gpu_name}")
        print(f"{check_mark(True)} GPU memory: {gpu_memory:.2f} GB")
        
        if gpu_memory < 4:
            print(f"\n{YELLOW}WARNING: GPU memory < 4GB. Consider reducing batch size.{RESET}")
    else:
        print(f"{check_mark(False)} CUDA not available")
        print(f"{YELLOW}WARNING: Experiments will run on CPU (very slow).{RESET}")
        print(f"\nRecommendations:")
        print(f"  1. Use SLURM cluster (12-24 hours with GPU)")
        print(f"  2. Use Baidu AI Studio (free V100)")
        print(f"  3. Install CUDA locally (if GPU present)")
    
    return cuda_available


def check_data(data_dir='./data/moleculenet'):
    """Check dataset files"""
    print_header("3. DATA CHECK")
    
    required_datasets = [
        'ESOL.csv', 'FreeSolv.csv', 'Lipo.csv',
        'BACE.csv', 'BBBP.csv', 'HIV.csv',
        'ClinTox.csv', 'Tox21.csv', 'SIDER.csv'
    ]
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"{check_mark(False)} Data directory not found: {data_dir}")
        print(f"\n{RED}Create data directory and add CSV files.{RESET}")
        return False
    
    all_ok = True
    
    for dataset in required_datasets:
        filepath = data_path / dataset
        
        if filepath.exists():
            # Check file format
            try:
                df = pd.read_csv(filepath)
                
                # Check for 'smiles' column
                if 'smiles' not in df.columns:
                    print(f"{check_mark(False)} {dataset:<15} Missing 'smiles' column")
                    all_ok = False
                    continue
                
                # Check number of samples
                n_samples = len(df)
                
                # Check number of tasks
                n_tasks = len(df.columns) - 1  # Exclude smiles column
                
                print(f"{check_mark(True)} {dataset:<15} {n_samples:>6} samples, {n_tasks} task(s)")
                
            except Exception as e:
                print(f"{check_mark(False)} {dataset:<15} Error reading: {str(e)}")
                all_ok = False
        else:
            print(f"{check_mark(False)} {dataset:<15} File not found")
            all_ok = False
    
    if not all_ok:
        print(f"\n{RED}Data check FAILED. Fix dataset files.{RESET}")
        print(f"\nRequired format:")
        print(f"  Column 1: smiles (SMILES strings)")
        print(f"  Columns 2+: Task labels")
    
    return all_ok


def check_directories():
    """Check/create required directories"""
    print_header("4. DIRECTORY CHECK")
    
    required_dirs = [
        './baselines/saved_models',
        './baselines/results',
        './logs'
    ]
    
    all_ok = True
    
    for dirname in required_dirs:
        dirpath = Path(dirname)
        
        if dirpath.exists():
            # Check if writable
            test_file = dirpath / '.write_test'
            try:
                test_file.touch()
                test_file.unlink()
                print(f"{check_mark(True)} {dirname:<30} Exists and writable")
            except:
                print(f"{check_mark(False)} {dirname:<30} Exists but NOT writable")
                all_ok = False
        else:
            # Create directory
            try:
                dirpath.mkdir(parents=True, exist_ok=True)
                print(f"{check_mark(True)} {dirname:<30} Created")
            except:
                print(f"{check_mark(False)} {dirname:<30} Failed to create")
                all_ok = False
    
    if not all_ok:
        print(f"\n{RED}Directory check FAILED. Fix permissions.{RESET}")
    
    return all_ok


def check_scripts():
    """Check if all required scripts exist"""
    print_header("5. SCRIPT CHECK")
    
    required_scripts = [
        'gnn_baselines.py',
        'aggregate_gnn_results.py',
        'generate_baseline_table.py'
    ]
    
    optional_scripts = [
        'run_gnn_baselines.sh',
        'run_gnn_baselines_local.bat'
    ]
    
    all_ok = True
    
    for script in required_scripts:
        if Path(script).exists():
            print(f"{check_mark(True)} {script:<40} Found")
        else:
            print(f"{check_mark(False)} {script:<40} NOT FOUND (required)")
            all_ok = False
    
    for script in optional_scripts:
        if Path(script).exists():
            print(f"{check_mark(True)} {script:<40} Found")
        else:
            print(f"{YELLOW}⚠{RESET} {script:<40} Not found (optional)")
    
    if not all_ok:
        print(f"\n{RED}Script check FAILED. Ensure all required scripts are present.{RESET}")
    
    return all_ok


def estimate_time():
    """Estimate experiment completion time"""
    print_header("6. TIME ESTIMATION")
    
    total_experiments = 3 * 9 * 5  # 3 models × 9 datasets × 5 seeds
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        
        if 'V100' in gpu_name or 'A100' in gpu_name:
            time_per_exp = "3-5 minutes"
            total_time = "4-8 hours"
            execution = "SLURM cluster (parallel)"
        elif 'GTX 1660' in gpu_name or 'RTX' in gpu_name:
            time_per_exp = "10-15 minutes"
            total_time = "24-36 hours (sequential)"
            execution = "Local GPU (sequential)"
        else:
            time_per_exp = "5-10 minutes"
            total_time = "12-24 hours"
            execution = "GPU (varies)"
    else:
        time_per_exp = "30-60 minutes"
        total_time = "3-7 days (sequential)"
        execution = "CPU (very slow)"
    
    print(f"Total experiments: {total_experiments}")
    print(f"Execution mode: {execution}")
    print(f"Time per experiment: {time_per_exp}")
    print(f"Estimated total time: {total_time}")
    
    if not torch.cuda.is_available():
        print(f"\n{YELLOW}WARNING: CPU execution is very slow. Strongly recommend using GPU.{RESET}")


def main():
    """Run all checks"""
    print(f"{BOLD}\n{'='*80}")
    print(f"Phase 1 GNN Baseline Experiments - Pre-Flight Checklist")
    print(f"{'='*80}{RESET}\n")
    
    # Run checks
    env_ok = check_environment()
    gpu_ok = check_gpu()
    data_ok = check_data()  # Update path if needed
    dir_ok = check_directories()
    script_ok = check_scripts()
    
    estimate_time()
    
    # Final verdict
    print_header("FINAL VERDICT")
    
    all_ok = env_ok and data_ok and dir_ok and script_ok
    
    if all_ok:
        print(f"{GREEN}{BOLD}✓ ALL CHECKS PASSED - Ready to start experiments!{RESET}\n")
        
        if torch.cuda.is_available():
            print(f"Recommended execution:")
            print(f"  sbatch run_gnn_baselines.sh  (SLURM cluster)")
            print(f"  OR")
            print(f"  python gnn_baselines.py --model GIN --dataset BBBP --seed 0  (test run)")
        else:
            print(f"{YELLOW}WARNING: No GPU detected. Consider using SLURM cluster or Baidu AI Studio.{RESET}")
        
    else:
        print(f"{RED}{BOLD}✗ CHECKS FAILED - Fix issues before starting experiments{RESET}\n")
        
        if not env_ok:
            print(f"  → Install missing Python packages")
        if not data_ok:
            print(f"  → Fix dataset files")
        if not dir_ok:
            print(f"  → Fix directory permissions")
        if not script_ok:
            print(f"  → Ensure all scripts are present")
    
    print()
    
    return all_ok


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
