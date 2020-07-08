import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from src.model import Net
from src.experiment_setup import ex
from src.utils import save_model


def compute_val(model, val_loader, criterion):
    model.eval()
    val_loss = 0.
    for _, data in enumerate(val_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # print statistics
        val_loss += loss.item()

    return val_loss / len(val_loader)


def load_data(args):
    if args.dataset == "CIFAR10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_val = torchvision.datasets.CIFAR10(root=args.root, train=True,
                                                 download=True, transform=transform)
    else:
        raise NotImplementedError

    # Train-Val split
    targets = train_val.targets
    train_idx, val_idx = train_test_split(
        range(len(targets)),
        test_size=args.valsize,
        shuffle=True, stratify=targets)

    train_set = torch.utils.data.Subset(train_val, train_idx)
    val_set = torch.utils.data.Subset(train_val, val_idx)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        sampler=None,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        sampler=None,
    )

    return train_loader, val_loader


@ex.automain
def main(_run):
    args = argparse.Namespace(**_run.config)

    # Load data
    train_loader, val_loader = load_data(args)

    # Define Network
    model = Net()
    print(model)

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    _run.info["steps_per_epoch"] = steps_per_epoch
    args.out_dir = "{}/{}/pretrained/".format(ex.observers[0].basedir, _run._id)
    os.makedirs(args.out_dir)
    args.tensorboard_dir = "{}/{}/tensorboard/".format(ex.observers[0].basedir, _run._id)
    os.makedirs(args.tensorboard_dir)
    writer = SummaryWriter(log_dir=args.tensorboard_dir)

    # Train the network
    for epoch in range(1, args.epochs + 1):  # loop over the dataset multiple times
        model.train()
        args.current_epoch = epoch
        train_loss_epoch = 0.

        running_loss = 0.
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_loss_epoch += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i + 1, running_loss / 2000))
                step = epoch * steps_per_epoch + i + 1
                _run.log_scalar("training.loss.step", running_loss / 2000, step)
                writer.add_scalar("Loss Steps/train", running_loss / 2000, step)
                running_loss = 0.

        # Save model
        if epoch % args.checkpoint == 0:
            save_model(args, model)
        # compute validation loss
        val_loss_epoch = compute_val(model, val_loader, criterion)
        _run.log_scalar("val.loss.epoch", val_loss_epoch)
        _run.log_scalar("train.loss.epoch", train_loss_epoch / steps_per_epoch)
        writer.add_scalar("Learning_rate", args.lr, epoch)
        writer.add_scalars("Loss vs Epoch", {"train_loss": train_loss_epoch / steps_per_epoch,
                                             "val_loss": val_loss_epoch}, epoch)

    print('Finished Training')
