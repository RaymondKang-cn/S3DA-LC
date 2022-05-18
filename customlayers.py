import torch.nn as nn
import config as config
import torch
from torchvision import models


# classifier
class Classifiers(nn.Module):

    def __init__(self, input_dim, domains, classes):
        super(Classifiers, self).__init__()
        self.domains = domains
        self.classes = classes
        self.input_dim = input_dim
        if domains == 2:
            self.linear1 = nn.Linear(self.input_dim, self.classes)
            self.linear2 = nn.Linear(self.input_dim, self.classes)
            self.linear = [self.linear1, self.linear2]
        elif domains == 3:
            self.linear1 = nn.Linear(self.input_dim, self.classes)
            self.linear2 = nn.Linear(self.input_dim, self.classes)
            self.linear3 = nn.Linear(self.input_dim, self.classes)
            self.linear = [self.linear1, self.linear2, self.linear3]
        elif domains == 5:
            self.linear1 = nn.Linear(self.input_dim, self.classes)
            self.linear2 = nn.Linear(self.input_dim, self.classes)
            self.linear3 = nn.Linear(self.input_dim, self.classes)
            self.linear4 = nn.Linear(self.input_dim, self.classes)
            self.linear5 = nn.Linear(self.input_dim, self.classes)
            self.linear = [self.linear1, self.linear2, self.linear3, self.linear4, self.linear5]

    def forward(self, x, state):
        logits = []
        if state == 'warm_up':
            x = x.chunk(self.domains)
            for i in range(self.domains):
                logits.append(self.linear[i](x[i]))
        elif state == 'eval':
            for i in range(self.domains):
                logits.append(self.linear[i](x))
        elif state == 'adapt':
            logits = [[],[]]
            x = x.chunk(self.domains+1)
            for i in range(self.domains):
                logits[0].append(self.linear[i](x[i]))
                logits[1].append(self.linear[i](x[self.domains]))
        return logits


# forward layer
class ForwardLayer(nn.Module):

    def __init__(self, inp_lin1, inp_lin2, f_dims):
        super(ForwardLayer, self).__init__()
        self.inp_lin1 = inp_lin1
        self.inp_lin2 = inp_lin2
        self.f_dims = f_dims

        self.net = nn.Sequential(
            nn.Linear(self.inp_lin1, self.inp_lin2),
            nn.ELU(),
            nn.Linear(self.inp_lin2, self.inp_lin2),
            nn.BatchNorm1d(self.inp_lin2),
            nn.ELU(),
            nn.Linear(self.inp_lin2, self.f_dims),
            nn.ELU(),
            nn.Linear(self.f_dims, self.f_dims),
            nn.BatchNorm1d(self.f_dims),
            nn.ELU()
        )

    def forward(self, x):
        return self.net(x)


# backbone layer
class BackBoneLayer(nn.Module):

    def __init__(self, pre, out_feats):
        super(BackBoneLayer, self).__init__()

        if pre == 'resnet101':
            temp_resnet = models.resnet101(pretrained=True)
            self.features = nn.Sequential(*[x for x in list(temp_resnet.children())[:-1]])
        elif pre == 'resnet50':
            temp_resnet = models.resnet50(pretrained=True)
            self.features = nn.Sequential(*[x for x in list(temp_resnet.children())[:-1]])

        self.pre = pre
        self.out_feats = out_feats

    def forward(self, x):
        return self.features(x).view((x.shape[0], self.out_feats))
