import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

class Trainer:
    def __init__(self,
                 lr,
                 batch_size,
                 n_epochs,
                 device,
                 optimizer=torch.optim.Adam):
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.cross_entropy = nn.CrossEntropyLoss()
        self.device = device

    def training_task(self, model, train_loader, head=0, verbose=True):
        model.train()
        task_size = len(train_loader.dataset)
        optimizer = self.optimizer(model.parameters(), lr=self.lr)
        if verbose:
          pbar = tqdm(total=self.n_epochs, desc='', position=0)
        for _ in range(self.n_epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = self.vcl_loss(model=model, inputs=x, targets=y, head=head, task_size=task_size)
                loss.backward()
                optimizer.step()
            if verbose:
              pbar.update(1)
        if verbose:
          pbar.close()
        model.update_prior(head)

    def initial_training(self, model, train_loader, head=0, verbose=True):
        model.train()
        task_size = len(train_loader.dataset)
        optimizer = self.optimizer(model.parameters(), lr=self.lr)
        if verbose:
          pbar = tqdm(total=self.n_epochs, desc='', position=0)
        for _ in range(self.n_epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                predictions = model.prediction(x, head, prob=False)
                loss = self.cross_entropy(predictions, y)
                loss.backward()
                optimizer.step()
            if verbose:
              pbar.update(1)
        if verbose:
          pbar.close()
        model.update_prior(head)

    def training_weightedset(self, model, train_loader, head=0, verbose=True):
        model.train()
        task_size = len(train_loader.dataset)
        optimizer = self.optimizer(model.parameters(), lr=self.lr)
        if verbose:
          pbar = tqdm(total=self.n_epochs, desc='', position=0)
        for _ in range(self.n_epochs):
            for x, y, weight in train_loader:
                x, y, weight = x.to(self.device), y.to(self.device), weight.to(self.device)
                optimizer.zero_grad()
                loss = self.vcl_weighted_loss(model=model, 
                                              inputs=x, 
                                              targets=y, 
                                              task_size=task_size, 
                                              weights=weight, 
                                              head=head)
                loss.backward()
                optimizer.step()
            if verbose:
              pbar.update(1)
        if verbose:
          pbar.close()
        model.update_prior(head)

    def training_multiple_tasks(self, model, train_loaders, heads, verbose=True):
        model.train()
        optimizer = self.optimizer(model.parameters(), lr=self.lr)
        for _ in range(self.n_epochs):
            order = torch.randperm(len(train_loaders))
            for idx in order:
                task_size = len(train_loaders[idx].dataset)
                for x, y in train_loaders[idx]:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    loss = self.vcl_loss(model=model, 
                                         inputs=x, 
                                         targets=y, 
                                         head=heads[idx], 
                                         task_size=task_size)
                    loss.backward()
                    optimizer.step()
        for head in heads:
            # if called by coreset, should not change anything
            model.update_prior(head)

    def training_multiple_weighted_tasks(self, model, train_loaders, heads, verbose=True):
        model.train()
        optimizer = self.optimizer(model.parameters(), lr=self.lr)
        for _ in range(self.n_epochs):
            order = torch.randperm(len(train_loaders))
            for idx in order:
                task_size = len(train_loaders[idx].dataset)
                for x, y, weight in train_loaders[idx]:
                    x, y, weight = x.to(self.device), y.to(self.device), weight.to(self.device)
                    optimizer.zero_grad()
                    loss = self.vcl_weighted_loss(model=model, 
                                                 inputs=x, 
                                                 targets=y, 
                                                 head=heads[idx],
                                                 weights=weight,
                                                 task_size=task_size)
                    loss.backward()
                    optimizer.step()
        for head in heads:
            # if called by coreset, should not change anything since the model is only used for evaluation
            model.update_prior(head)


    
    def loss(self, model, inputs, targets, head=0, n_samples=10):
        outputs = torch.cat([model.one_sample_forward(inputs, head) for _ in range(n_samples)], dim=0)
        return self.cross_entropy(outputs, targets.repeat(n_samples))
        
    def vcl_loss(self, model, inputs, targets, task_size, head=0, n_samples=10):
        return model.kl_divergence(head) / task_size + self.loss(model, inputs, targets, head=head, n_samples=n_samples)

    def vcl_weighted_loss(self, model, inputs, targets, task_size, weights, head=0, n_samples=10):
        outputs = torch.cat([model.one_sample_forward(inputs, head) for _ in range(n_samples)], dim=0)
        weighted_loss = (weights.repeat(n_samples) * nn.CrossEntropyLoss(reduction='none')(outputs, targets.repeat(n_samples))).mean()
        return model.kl_divergence(head) / task_size + weighted_loss
        