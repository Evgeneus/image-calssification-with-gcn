import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse

from src.model import Net
from src.experiment_setup import ex


@ex.automain
def main(_run):
    args = argparse.Namespace(**_run.config)
    train_sampler = None

    if args.dataset == "CIFAR10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root=args.root, train=True,
                                                download=True, transform=transform)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

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
    # Train the network
    for epoch in range(args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
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
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                step = epoch * steps_per_epoch + i + 1
                _run.log_scalar("training.loss", running_loss / 2000, step)
                running_loss = 0.0

    print('Finished Training')
