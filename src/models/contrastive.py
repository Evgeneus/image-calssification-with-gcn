import torch
import torch.nn as nn
from itertools import combinations
import numpy as np
from models.resnet import ResNet18


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ContrastiveNet(nn.Module):
    def __init__(self, batch_size, temperature, device, projection_dim, pretrained_encoder):
        super(ContrastiveNet, self).__init__()
        # load or create encoder
        if pretrained_encoder is not None:
            self.encoder = ResNet18(output_layer=True)
            self.encoder.load_state_dict(torch.load(pretrained_encoder, map_location=device))
            self.encoder.linear = Identity()
        else:
            self.encoder = ResNet18(output_layer=False)
        embed_dim = 512  # output from ResNet18

        # We use a MLP with one hidden layer to obtain projection z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim, bias=False),
        )

        self.num_neg = 10
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def _cosine_similarity(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)

        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

    def compute_logits(self, similarity, labels):
        labels = labels.cpu().data.numpy()
        logits = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            for anchor, positive in anchor_positives:
                sim_positives = similarity[anchor][positive]
                sim_negatives = similarity[anchor][np.random.choice(negative_indices, self.num_neg, replace=False)]
                l = torch.cat((sim_positives.unsqueeze(0), sim_negatives), dim=0)
                logits.append(l)

        return torch.stack(logits, dim=0)

    def forward(self, x, targets):
        # make forward pass
        num_classes = targets.unique().size()[0]

        x = self.encoder(x)
        x = self.projector(x)

        # one hot encoding buffer
        y_onehot = torch.zeros(self.batch_size, num_classes)
        y_onehot.scatter_(1, targets.unsqueeze(1), 1)

        # compute similarity score (cosine sim)
        sim = self._cosine_similarity(x) / self.temperature

        # compute pairs of positives and random negatives
        logits = self.compute_logits(sim, targets)
        labels = torch.zeros(logits.shape[0]).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss
