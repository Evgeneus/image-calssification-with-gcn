import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


def load_data(args):
    if args.dataset == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_val = torchvision.datasets.CIFAR10(root=args.root, train=True,
                                                 download=True, transform=transform_train)
    else:
        raise NotImplementedError

    # Train-Val split
    targets = train_val.targets
    train_idx, val_idx = train_test_split(
        range(len(targets)), test_size=args.valsize,
        shuffle=True, stratify=targets, random_state=args.seed)

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


def load_data_test(args):
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
