"""
Sacred experiment file
"""

from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch


ex = Experiment("Classification")


@ex.config
def config():
    config_file = "../config/config_supervised.yaml"
    ex.add_config(config_file)

    # file output directory
    ex.observers.append(FileStorageObserver("../logs/classification"))

    ex.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ex.n_gpu = torch.cuda.device_count()


# config for contrastive learning
ex_contrastive = Experiment("Contrastive")

@ex_contrastive.config
def config_contrastive():
    config_file = "../config/config_contrastive.yaml"
    ex_contrastive.add_config(config_file)

    # file output directory
    ex_contrastive.observers.append(FileStorageObserver("../logs/contrastive"))

    ex_contrastive.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ex_contrastive.n_gpu = torch.cuda.device_count()
