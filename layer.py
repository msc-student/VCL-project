import torch
from torch import nn
from torch.nn.parameter import Parameter

class GaussianMeanFieldLayer(nn.Module):
    def __init__(self, input_dim, output_dim, initial_sigma=torch.tensor(1e-3)):
        super(GaussianMeanFieldLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initial_sigma = initial_sigma

        # Prior
        self.register_buffer('prior_weight_mu', torch.zeros(self.output_dim, self.input_dim))
        self.register_buffer('prior_weight_sigma', torch.zeros(self.output_dim, self.input_dim))
        self.register_buffer('prior_bias_mu', torch.zeros(self.output_dim))
        self.register_buffer('prior_bias_sigma', torch.zeros(self.output_dim))

        # Posterior
        self.posterior_weight_mu = Parameter(torch.zeros(self.output_dim, self.input_dim), requires_grad=True)
        self.posterior_weight_sigma = Parameter(torch.zeros(self.output_dim, self.input_dim), requires_grad=True)
        self.posterior_bias_mu = Parameter(torch.zeros(self.output_dim), requires_grad=True)
        self.posterior_bias_sigma = Parameter(torch.zeros(self.output_dim), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        # initialise the means with a normal distribution
        torch.nn.init.normal_(self.posterior_weight_mu, mean=0., std=0.1)
        torch.nn.init.normal_(self.posterior_bias_mu, mean=0., std=0.1)
        # initialise variance with a fixed value
        torch.nn.init.constant_(self.posterior_weight_sigma, torch.log(self.initial_sigma))
        torch.nn.init.constant_(self.posterior_bias_sigma, torch.log(self.initial_sigma))

    def sample_parameters(self):
        # Reparametrization trick
        weights = self.posterior_weight_mu + torch.randn_like(self.posterior_weight_mu) * torch.exp(0.5*self.posterior_weight_sigma)
        bias = self.posterior_bias_mu + torch.randn_like(self.posterior_bias_mu) * torch.exp(0.5*self.posterior_bias_sigma)
        return weights, bias
    
    def update_prior(self):
        self.prior_weight_mu.data.copy_(self.posterior_weight_mu.data)
        self.prior_weight_sigma.data.copy_(self.posterior_weight_sigma.data)
        self.prior_bias_mu.data.copy_(self.posterior_bias_mu.data)
        self.prior_bias_sigma.data.copy_(self.posterior_bias_sigma.data)
        

    def kl_divergence(self, epsilon=1e-9):
        # all elements of the posterior and prior are independent, 
        # so the kl divergence for all the parameters is the sum of the kl divergence for each parameter
        weight_kldiv = (torch.exp(self.posterior_weight_sigma - self.prior_weight_sigma) 
                         + (self.prior_weight_mu - self.posterior_weight_mu)**2/(torch.exp(self.prior_weight_sigma) + epsilon)
                         - 1 + (self.prior_weight_sigma - self.posterior_weight_sigma))/2
        bias_kldiv = (torch.exp(self.posterior_bias_sigma)/torch.exp(self.prior_bias_sigma) 
                         + (self.prior_bias_mu - self.posterior_bias_mu)**2/(torch.exp(self.prior_bias_sigma) + epsilon)
                         - 1 + (self.prior_bias_sigma - self.posterior_bias_sigma))/2
        return weight_kldiv.sum() + bias_kldiv.sum()
    
    def forward(self, x):
        weights, bias = self.sample_parameters()
        return nn.functional.linear(x, weights, bias)

    def prediction(self, x):
        return nn.functional.linear(x, self.posterior_weight_mu, self.posterior_bias_mu)