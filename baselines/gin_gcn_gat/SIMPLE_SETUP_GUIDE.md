# GNN Baselines - SIMPLE SETUP GUIDE

**5 Simple Steps to Run All Experiments**

---

## STEP 1: Create Folders

Open Command Prompt (Windows Key + R, type `cmd`, press Enter)

Copy and paste these commands one by one:

```cmd
cd D:\molprop_project
mkdir baselines
cd baselines
mkdir gin_gcn_gat
cd gin_gcn_gat
mkdir baselines
cd baselines
mkdir saved_models
mkdir results
cd ..
mkdir logs
```

**What you just created:**
```
D:\molprop_project\baselines\gin_gcn_gat\
├── baselines\
│   ├── saved_models\
│   └── results\
└── logs\
```

---

## STEP 2: Copy Downloaded Files

You downloaded 4 Python files. Copy them ALL to:
```
D:\molprop_project\baselines\gin_gcn_gat\
```

**The 4 files are:**
1. `gnn_baselines.py` (22 KB)
2. `run_all_experiments.py` (5 KB)
3. `aggregate_gnn_results.py` (7 KB)
4. `generate_baseline_table.py` (11 KB)

**How to copy:**
- Open your Downloads folder
- Select all 4 files
- Right-click → Copy
- Go to `D:\molprop_project\baselines\gin_gcn_gat\`
- Right-click → Paste

---

## STEP 3: Edit Data Path

Open `run_all_experiments.py` with Notepad:
- Right-click `run_all_experiments.py`
- Choose "Edit with Notepad" (or "Open with → Notepad")

**Find line 18** (it looks like this):
```python
DATA_DIR = r'D:\molprop_project\data\moleculenet'
```

**Change it to match WHERE YOUR CSV FILES ARE.**

For example, if your ESOL.csv is at:
```
D:\molprop_project\temp\chemprop_data\ESOL.csv
```

Then change line 18 to:
```python
DATA_DIR = r'D:\molprop_project\temp\chemprop_data'
```

**Save and close Notepad** (File → Save)

---

## STEP 4: Find Your Conda Environment Name

Open Command Prompt and type:

```cmd
conda env list
```

You'll see something like:
```
base                  *  C:\Users\...\anaconda3
pytorch                  C:\Users\...\anaconda3\envs\pytorch
```

**Write down the name** of the environment where you have PyTorch installed.

Common names:
- `base` (most common)
- `pytorch`
- `torch`
- `py310`

If you're not sure, it's probably `base`.

---

## STEP 5: Run Experiments

Open Command Prompt and type these commands:

```cmd
cd D:\molprop_project\baselines\gin_gcn_gat

conda activate base

python run_all_experiments.py
```

**Replace `base` with YOUR environment name from Step 4!**

---

## THAT'S IT!

Press Enter when it asks "Press Enter to start..."

The script will:
- Run all 135 experiments (this takes 2-3 days)
- Show progress after each experiment
- Save results automatically
- Create log files for debugging

**You can close the Command Prompt window**, but the experiments will stop.
**Leave your computer on** and let it run overnight.

---

## After Experiments Finish

When all 135 experiments are done, run this to see results:

```cmd
cd D:\molprop_project\baselines\gin_gcn_gat

conda activate base

python aggregate_gnn_results.py
```

This will show you a nice table comparing GIN vs GCN vs GAT vs D-MPNN.

---

## Common Problems

### Problem: "conda not recognized"
**Solution:** You need to use Anaconda Prompt instead of regular Command Prompt
- Search for "Anaconda Prompt" in Windows Start menu
- Use that instead

### Problem: "No module named torch"
**Solution:** Wrong environment activated
- Try: `conda activate base`
- Or try: `conda activate pytorch`
- Or install PyTorch: `pip install torch`

### Problem: "FileNotFoundError: ESOL.csv"
**Solution:** Wrong data path in Step 3
- Go back to Step 3
- Find where ESOL.csv actually is
- Update the path correctly

### Problem: Experiment crashes/fails
**Solution:** Check the log file
- Open `logs\GIN_ESOL_seed0.log` (or whichever failed)
- Look at the error message
- Share it with me and I'll help

---

## Quick Test (Before Running All 135)

Want to test if everything works before running all experiments?

Try this (runs just ONE experiment as a test):

```cmd
cd D:\molprop_project\baselines\gin_gcn_gat

conda activate base

python gnn_baselines.py --model GIN --dataset BBBP --seed 0 --data_dir D:\molprop_project\data\moleculenet --save_dir baselines\saved_models --results_file baselines\results\test.json
```

If this works (takes 5-10 minutes), then everything is set up correctly!

---

## Files After Completion

After experiments finish, you'll have:

```
D:\molprop_project\baselines\gin_gcn_gat\
├── gnn_baselines.py              (your code)
├── run_all_experiments.py        (your code)
├── aggregate_gnn_results.py      (your code)
├── generate_baseline_table.py    (your code)
│
├── baselines\
│   ├── saved_models\
│   │   ├── GIN_ESOL_seed0_best.pt
│   │   ├── GIN_ESOL_seed1_best.pt
│   │   └── ... (135 model files)
│   └── results\
│       └── gnn_results.json      ← ALL YOUR RESULTS HERE
│
└── logs\
    ├── GIN_ESOL_seed0.log
    ├── GIN_ESOL_seed1.log
    └── ... (135 log files)
```

The most important file is `baselines\results\gnn_results.json` - this has all your experimental results!

---

**Questions? Just ask!** But try Step 1-5 first and tell me which step fails.
