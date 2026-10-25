"""Paths used by every script. Override with environment variables:

    DATA_DIR     where clue.zip lives                      (default: <repo>/data)
    WORK_DIR     intermediate files (parquet, base tables) (default: <repo>/work)
    RESULTS_DIR  experiment outputs (json, logs)           (default: <repo>/results)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('DATA_DIR', os.path.join(ROOT, 'data'))
WORK = os.environ.get('WORK_DIR', os.path.join(ROOT, 'work'))
RESULTS = os.environ.get('RESULTS_DIR', os.path.join(ROOT, 'results'))

ZIP = os.path.join(DATA, 'clue.zip')                    # CLUE-LDS from Zenodo 7119953
PARQUET = os.path.join(WORK, 'events.parquet')          # step01_data_to_parquet.py
STATS = os.path.join(WORK, 'user_stats.json')           # step02_user_stats.py
BASE = os.path.join(WORK, 'base')                       # step04_daily_tables.py
K3 = os.path.join(WORK, 'k3')                           # K3 window tables (step10/step11)
A_DIR = os.path.join(WORK, 'A')                         # baseline A runs (step05_model_A_landauer.py)
A_DUMP = os.path.join(A_DIR, 'seed1_t0.6.jsonl')        # per-day decisions of A, seed 1
A_LABELS = os.path.join(A_DIR, 'seed1', 'labels.txt')   # uid-swap labels, seed 1

for d in (WORK, RESULTS, K3, A_DIR):
    os.makedirs(d, exist_ok=True)
