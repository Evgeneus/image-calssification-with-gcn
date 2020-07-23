import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from experiment_setup import ex
from dataloader import load_data
from utils import save_model, CrossEntropyLossSoft


best_val_acc = 0.


def compute_val(model, val_loader, args):
    global best_val_acc
    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_loss = 0.
    correct = 0
    total = 0
    with torch.no_grad():
        for _, data in enumerate(val_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(args.device), data[1].to(args.device)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # print statistics
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print("Val acc: {}".format(acc))

    if acc > best_val_acc:
        print("Saving model...")
        best_val_acc = acc
        save_model(args, model, 'best')

    return val_loss / len(val_loader)


def load_model(args):
    if args.model == "ResNet":
        from models.resnet import ResNet18
        model = ResNet18(output_layer=True, num_classes=10)
    elif args.model == "GraphResNet":
        from models.graph_resnet import GraphResNet
        model = GraphResNet(args)
    else:
        raise NotImplementedError

    model = model.to(args.device)

    return model


@ex.automain
def main(_run):
    args = argparse.Namespace(**_run.config)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader = load_data(args)

    # Define Network
    model = load_model(args)
    if args.device.type == "cuda":
        model = model.to(args.device)
        import torch.backends.cudnn as cudnn
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # Define a Loss function and optimizer
    criterion = CrossEntropyLossSoft()

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
            inputs, labels = data[0].to(args.device), data[1].to(args.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, labels_soft = model(inputs, labels)

            loss = criterion(outputs, labels_soft)

            loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_loss_epoch += loss.item()

            k = 20
            if i % k == 0:  # print every k mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i + 1, running_loss / k))
                step = epoch * steps_per_epoch + i + 1
                _run.log_scalar("training.loss.step", running_loss / k, step)
                writer.add_scalar("Loss Steps/train", running_loss / k, step)
                running_loss = 0.

        # Save model
        if epoch % args.checkpoint == 0:
            save_model(args, model)
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
