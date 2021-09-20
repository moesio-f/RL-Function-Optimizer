import os

SRC_DIR = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.normpath(os.path.join(SRC_DIR, '..'))
EXPERIMENTS_DIR = os.path.normpath(os.path.join(ROOT_DIR, 'experiments'))
POLICIES_DIR = os.path.normpath(os.path.join(EXPERIMENTS_DIR, 'policies'))
BASELINES_DIR = os.path.normpath(os.path.join(EXPERIMENTS_DIR, 'baselines'))
TRAINING_DIR = os.path.normpath(os.path.join(EXPERIMENTS_DIR, 'training'))
