from abc import ABC
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)


class Discriminator(nn.Module, ABC):
    def __init__(self, n_states, n_skills, n_hidden_filters=256):
        super(Discriminator, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
        init_weight(self.q, initializer="xavier uniform")
        self.q.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        logits = self.q(x)
        return logits


class ValueNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_hidden_filters=256):
        super(ValueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.value, initializer="xavier uniform")
        self.value.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        return self.value(x)


class QvalueNetwork(nn.Module, ABC):
    """
    Q(s,a,z) – returns a vector with |A| elements.
    """
    def __init__(self, n_states, n_actions, n_hidden_filters=256):
        super().__init__()
        self.h1 = nn.Linear(n_states, n_hidden_filters); init_weight(self.h1)
        self.h2 = nn.Linear(n_hidden_filters, n_hidden_filters); init_weight(self.h2)
        self.q  = nn.Linear(n_hidden_filters, n_actions)
        init_weight(self.q, initializer="xavier uniform")

    def forward(self, s):
        x = F.relu(self.h1(s))
        x = F.relu(self.h2(x))
        return self.q(x)                          #  shape (B, |A|)


class PolicyNetwork(nn.Module, ABC):
    """
    Stochastic policy π(a|s,z)  ——  discrete actions
    """
    def __init__(self, n_states, n_actions, n_hidden_filters=256):
        super().__init__()
        self.n_states   = n_states
        self.n_actions  = n_actions
        self.n_hid      = n_hidden_filters

        self.h1 = nn.Linear(n_states,      n_hidden_filters); init_weight(self.h1)
        self.h2 = nn.Linear(n_hidden_filters, n_hidden_filters); init_weight(self.h2)
        self.logits = nn.Linear(n_hidden_filters, n_actions)
        init_weight(self.logits, initializer="xavier uniform")

    def forward(self, s):
        x = F.relu(self.h1(s))
        x = F.relu(self.h2(x))
        logits = self.logits(x)
        probs  = F.softmax(logits, dim=-1)             # π(a|s)
        log_p  = F.log_softmax(logits, dim=-1)
        return probs, log_p

    def sample(self, s, greedy=False):
        probs, log_p = self.forward(s)
        dist  = torch.distributions.Categorical(probs)
        a_idx = probs.argmax(-1) if greedy else dist.sample()
        return a_idx, log_p.gather(-1, a_idx.unsqueeze(-1))   # log π(a|s)