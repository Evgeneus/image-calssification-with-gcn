import argparse
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from experiment_setup import ex_contrastive
from dataloader import load_data, load_data_test


def load_model(args):
    if args.model == "ContrastiveResNet":
        from models.contrastive import ContrastiveNet
        model = ContrastiveNet(args)
        model.load_state_dict(torch.load(args.test_model, map_location=args.device))
    else:
        raise NotImplementedError
    model = model.to(args.device)

    return model


@ex_contrastive.automain
@torch.no_grad()
def main(_run):
    args = argparse.Namespace(**_run.config)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader = load_data(args)
    test_loader = load_data_test(args)

    # Define Network
    model = load_model(args)
    model.eval()

    # compute embeddings for Train data
    train_embeddings = []
    train_labels = []
    print("Computing Train embeddings..")
    for i, data_train in enumerate(train_loader, 0):
        print("{}/{}".format(i, len(train_loader)))
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data_train[0].to(args.device), data_train[1].to(args.device)
        embed = model.projector(model.encoder(inputs)).cpu()
        # l2 normalization to use cosine dist as equivalent to l2 dist in KNN
        embed = F.normalize(embed, dim=1, p=2).tolist()
        train_embeddings += embed
        train_labels += labels.cpu().tolist()
    print("Done!")

    # compute embeddings for Test data
    test_embeddings = []
    test_labels = []
    print("Computing Test embeddings..")
    for i, data_test in enumerate(test_loader, 0):
        print("{}/{}".format(i, len(test_loader)))
        # get the inputs; data is a list of [inputs, labels]
        embed = model.projector(model.encoder(inputs)).cpu()
        # l2 normalization to use cosine dist as equivalent to l2 dist in KNN
        embed = F.normalize(embed, dim=1, p=2).tolist()
        test_embeddings += embed
        test_labels += labels.cpu().tolist()
    print("Done!")

    print("Fit KNN classifier")
    knn = KNeighborsClassifier(n_neighbors=args.n_neighbors, n_jobs=args.workers)
    knn.fit(train_embeddings, train_labels)
    print("Done!")

    print("Predict Test Accuracy!")
    predicted_labels = knn.predict(test_embeddings)
    accuracy_test = accuracy_score(test_labels, predicted_labels)
    print("Test Accuracy: {}".format(accuracy_test))
    print("Done!")
