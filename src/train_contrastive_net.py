import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from experiment_setup import ex_contrastive
from dataloader import load_data


@torch.no_grad()
def compute_val(model, val_loader, args):
    model.eval()
    val_loss = 0.
    for _, data in enumerate(val_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(args.device), data[1].to(args.device)
        # forward
        loss = model(inputs, labels)
        val_loss += loss.item()
    print("Val loss: {}".format(val_loss / len(val_loader)))

    return val_loss / len(val_loader)


def load_model(args):
    if args.model == "ContrastiveResNet":
        from models.contrastive import ContrastiveNet
        model = ContrastiveNet(args)
    else:
        raise NotImplementedError
    model = model.to(args.device)

    return model


@ex_contrastive.automain
def main(_run):
    args = argparse.Namespace(**_run.config)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader = load_data(args)

    # Define Network
    model = load_model(args)
    print(model)

    # Define optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    _run.info["steps_per_epoch"] = steps_per_epoch
    args.out_dir = "{}/{}/pretrained/".format(ex_contrastive.observers[0].basedir, _run._id)
    os.makedirs(args.out_dir)
    args.tensorboard_dir = "{}/{}/tensorboard/".format(ex_contrastive.observers[0].basedir, _run._id)
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
            inputs, labels = data[0].to(args.device), data[1].to(args.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = model(inputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_loss_epoch += loss.item()
            k = 1
            if i % k == 0:  # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i + 1, running_loss / k))
                step = epoch * steps_per_epoch + i + 1
                _run.log_scalar("training.loss.step", running_loss / k, step)
                writer.add_scalar("Loss Steps/train", running_loss / k, step)
                running_loss = 0.

        # compute validation loss
        val_loss_epoch = compute_val(model, val_loader, args)
        _run.log_scalar("val.loss.epoch", val_loss_epoch)
        _run.log_scalar("train.loss.epoch", train_loss_epoch / steps_per_epoch)
        writer.add_scalar("Learning_rate", args.lr, epoch)
        writer.add_scalars("Loss vs Epoch", {"train_loss": train_loss_epoch / steps_per_epoch,
                                             "val_loss": val_loss_epoch}, epoch)
        print("epoch: {}".format(epoch))
        print("val.loss.epoch", val_loss_epoch)
        print("train.loss.epoch", train_loss_epoch / steps_per_epoch)

    print('Finished Training')
