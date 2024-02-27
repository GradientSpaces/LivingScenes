"""
May the Force be with you.
Main program 2019.3
Update 2020.10
Update 2021.7
"""

# import open3d as o3d  # resolve open3d and pytorch conflict
# import pyrender  # solve the wired pyrender - pytorch conflict
from dataset import get_dataset
from logger import Logger
from core.models import get_model
from core import solver_dict
from init import get_cfg, setup_seed
# import wandb

# wandb.init(project='VN_ILoc', sync_tensorboard=True)

# preparer configuration
cfg = get_cfg()

# set random seed
setup_seed(cfg["rand_seed"])

# prepare dataset
DatasetClass = get_dataset(cfg)
datasets_dict = dict()
for mode in cfg["modes"]:
    datasets_dict[mode] = DatasetClass(cfg, mode=mode)

# prepare models
ModelClass = get_model(cfg["model"]["model_name"])
model = ModelClass(cfg)

# prepare logger
logger = Logger(cfg)

# register dataset, models, logger to the solver
solver = solver_dict[cfg["runner"].lower()](cfg, model, datasets_dict, logger)

# optimize
solver.run()
