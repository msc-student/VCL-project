import torch
from torch import nn
import math
from layer import GaussianMeanFieldLayer
import torch.nn.functional as F


class DiscriminativeModel(nn.Module):
    def __init__(self,
                 input_dim=784,
                 n_hidden_layers=2,
                 hidden_dim=100,
                 output_dim=10,
                 n_heads=1,
                initial_sigma=1e-3):
        super(DiscriminativeModel, self).__init__()
        self.input_dim = input_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if isinstance(output_dim, list) else [output_dim]*n_heads
        self.n_heads = n_heads
        self.initial_sigma = torch.tensor(initial_sigma)
        self.layers = nn.Sequential(GaussianMeanFieldLayer(input_dim, hidden_dim, initial_sigma=self.initial_sigma), nn.ReLU(),
                                    *[GaussianMeanFieldLayer(hidden_dim, hidden_dim, initial_sigma=self.initial_sigma), nn.ReLU()]*(n_hidden_layers-1))
        self.heads = nn.ModuleList([GaussianMeanFieldLayer(self.hidden_dim, 
                                                           self.output_dim[n_head], 
                                                           initial_sigma=self.initial_sigma) for n_head in range(self.n_heads)])
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.kl_div_loss = nn.KLDivLoss()
    
    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.heads:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def update_prior(self, head=0):
        for layer in self.layers:
            if isinstance(layer, GaussianMeanFieldLayer):
                layer.update_prior()
        assert isinstance(self.heads[head], GaussianMeanFieldLayer) == True
        self.heads[head].update_prior()
    
    def kl_divergence(self, head=0, epsilon=1e-9):
        kl_div = 0
        for layer in self.layers:
            if isinstance(layer, GaussianMeanFieldLayer):
                kl_div += layer.kl_divergence(epsilon)
        assert isinstance(self.heads[head], GaussianMeanFieldLayer) == True
        kl_div += self.heads[head].kl_divergence(epsilon)
        return kl_div

    def get_features(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def one_sample_forward(self, x, head=0, log_output=False):
        for layer in self.layers:
            x = layer(x)
        output = self.heads[head](x)
        if log_output:
            return self.log_softmax(output)
        return output

    def forward(self, x, head=0, n_samples=10, log_output=False):
        output = torch.stack([self.one_sample_forward(x, head, log_output) for _ in range(n_samples)], dim=0)
        return output.mean(dim=0)

    def prediction(self, x, head=0, prob=True):
        for layer in self.layers:
            if isinstance(layer, GaussianMeanFieldLayer):
                x = layer.prediction(x)
            else:
                x = layer(x)
        output = self.heads[head](x)
        if prob:
            return self.softmax(output)
        else:
            return output
        