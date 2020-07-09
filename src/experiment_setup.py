"""
Sacred experiment file
"""

from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch


ex = Experiment("Cls")


@ex.config
def config():
    config_file = "../config/config.yaml"
    ex.add_config(config_file)

    # file output directory
    ex.observers.append(FileStorageObserver("../logs"))

    ex.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ex.n_gpu = torch.cuda.device_count()

