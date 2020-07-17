import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from src.experiment_setup import ex
from src.utils import load_model


def load_data(args):
    if args.dataset == "CIFAR10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test_set = torchvision.datasets.CIFAR10(root=args.root, train=False,
                                                download=True, transform=transform)
    else:
        raise NotImplementedError

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.workers,
        sampler=None,
    )

    return test_loader


def load_model_bold(args):
    if args.model == "ResNet":
        from models.model import ResNet
        model = ResNet()
    elif args.model == "ResNetGCN":
        from models.model import ResNetGCN
        model = ResNetGCN(args.graph_type)
    elif args.model == "ResNetGATCN":
        from models.model import ResNetGATCN
        model = ResNetGATCN(args.graph_type)
    else:
        raise NotImplementedError

    model = model.to(args.device)

    return model


@ex.automain
def main(_run):
    args = argparse.Namespace(**_run.config)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load test data
    test_loader = load_data(args)
    ex.info["test_size"] = len(test_loader.dataset)

    # Load model
    model_bold = load_model_bold(args)
    model = load_model(args, model_bold)
    model = model.to(args.device)
    model.eval()

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    running_loss = 0.
    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(args.device), data[1].to(args.device)

            # predict labels
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    loss = running_loss / len(test_loader)
    ex.info["accuracy"] = accuracy
    ex.info["loss"] = loss
    print("Testing the network on the {} test images".format(ex.info["test_size"]))
    print("Test Accuracy: {}".format(accuracy))
    print("Test Loss: {}".format(loss))
