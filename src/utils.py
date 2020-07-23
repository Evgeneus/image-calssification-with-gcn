import torch
from torch import nn


def save_model(args, model):
    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    out = args.out_dir + "checkpoint_{}.tar".format(args.current_epoch)
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)


def load_model(args, model, msg=None):
    if msg:
        model_path = "../logs/classification/{}/pretrained/model_{}.tar".format(args.experiment_id, msg)
    else:
        model_path = "../logs/classification/{}/pretrained/checkpoint_{}.tar".format(args.experiment_id, args.checkpoin_to_test)
    model.load_state_dict(torch.load(model_path))

    return model


class CrossEntropyLossSoft(nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLossSoft, self).__init__()
        self.weight = weight

    def forward(self, pred, soft_targets):
        logsoftmax = nn.LogSoftmax()
        if self.weight is not None:
            return torch.mean(torch.sum(- soft_targets * self.weight * logsoftmax(pred), 1))
        else:
            return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
