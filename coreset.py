import torch
from torch.utils.data import Subset, ConcatDataset, Dataset, DataLoader
import copy
from trainer import Trainer
import acquisition_scores
import numpy as np

class Coreset:
    def __init__(self, coreset_method, coreset_size, n_epochs, device, importance_sampler=None):
        self.coreset_method = coreset_method
        self.coreset_size = coreset_size
        self.trainer = Trainer(lr=0.001,
                                batch_size=coreset_size,
                                n_epochs=n_epochs,
                                device=device)
        self.coresets = []
        self.heads = []
        self.importance_sampler = importance_sampler

    def get_indices2(self, train_dataset):
        if (self.coreset_method is None):
            self.coresets.append(None)
            return train_dataset
        if (self.coreset_method == 'importance'):
            coresets = []
            for i, dataset in enumerate(train_dataset):
                coreset = self.importance_sampler.get_coreset(model, 
                                                             dataset,
                                                             self.coreset_size,
                                                             heads[i])
                coresets.append(coreset)
            self.coresets.append(coresets)
        if self.coreset_method == 'random':
            core_set, train_set = torch.utils.data.random_split(train_dataset, 
                                                                   [self.coreset_size, max(len(train_dataset)-self.coreset_size, 0)])
            self.coresets.append(core_set)
            return train_set


    def get_indices(self, train_dataset, model=None, head=None):
        if (self.coreset_method is None):
            self.coresets.append(None)
            return train_dataset
        if (self.coreset_method == 'importance'):
            print('Importance method')
            coreset = self.importance_sampler.get_coreset(model, 
                                                         train_dataset,
                                                         self.coreset_size,
                                                         head)
            self.coresets.append(coreset)
            dataset_idx = set(np.arange(len(train_dataset)))
            coreset_idx = set(coreset.indices.cpu().numpy())
            indices_t = torch.tensor(list(dataset_idx - coreset_idx)).to(self.trainer.device)
            return Subset(train_dataset, indices_t)
        if self.coreset_method == 'random':
            print('Random method')
            core_set, train_set = torch.utils.data.random_split(train_dataset, 
                                                                   [self.coreset_size, max(len(train_dataset)-self.coreset_size, 0)])
            self.coresets.append(core_set)
            return train_set
        

    def train_importance(self, initial_model, train_datasets, heads):
        if (self.coreset_method is None) or (self.coreset_size == 0) or (len(train_datasets) == 0):
            print('No coreset')
            return initial_model
        model = copy.deepcopy(initial_model)
        print('Training Coreset')
        if isinstance(train_datasets, list):
            dataloaders = [DataLoader(train_dataset, self.trainer.batch_size, shuffle=True) for train_dataset in train_datasets]
            self.trainer.training_multiple_weighted_tasks(model=model, train_loaders=dataloaders, heads=heads)
        else:
            dataloader = DataLoader(train_datasets, self.trainer.batch_size, shuffle=True)
            self.trainer.training_weightedset(model=model, train_loader=dataloader, heads=heads, verbose=False)
        return model

    def train(self, initial_model, train_datasets, heads):
        if (self.coreset_method is None) or (self.coreset_size == 0) or (len(train_datasets) == 0):
            print('No coreset')
            return initial_model
        model = copy.deepcopy(initial_model)
        if isinstance(train_datasets, list):
            dataloaders = [DataLoader(train_dataset, self.trainer.batch_size, shuffle=True) for train_dataset in train_datasets]
            print(f'Training Coreset')
            self.trainer.training_multiple_tasks(model=model, train_loaders=dataloaders, heads=heads)
        else:
            dataloader = DataLoader(train_datasets, self.trainer.batch_size, shuffle=True)
            self.trainer.training_task(model=model, train_loader=dataloader, head=heads)
        return model



class WeightedDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.weights = torch.zeros(len(self.dataset))

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y, self.weights[index]

    def __len__(self):
        return len(self.dataset)


class ImportanceSampledCoreset(torch.utils.data.Subset):
    def __init__(self, dataset, idxs, weights):
        self.dataset = dataset
        self.dataset.weights[idxs] = weights.cpu()
        self.indices = idxs

    def __getitem__(self, index):
        index_ = self.indices[index]
        return self.dataset[index_]

    def __len__(self):
        return len(self.indices)


class ImportanceSampling:
    def __init__(self, proposal, acquisition_function, batch_size, n_samples, temperature=1.0):
        self.proposal = proposal
        self.acquisition_function = acquisition_function
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.temperature = temperature

    def get_coreset(self, 
                    model, 
                    dataset,
                    coreset_size,
                   head=0):
        if len(dataset) == 0:
            return dataset
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        idxs, masses = self.get_idx_sampling(model,
                                             data_loader,
                                            coreset_size,
                                            head)
        return ImportanceSampledCoreset(WeightedDataset(dataset), idxs, masses)

    def get_idx_sampling(self,
                         model,
                         data_loader,
                        coreset_size,
                        head):
        scores = self.acquisition_function(model, data_loader, self.n_samples, head)
        probability_masses = scores / torch.sum(scores)
        if self.proposal == "proportional":
            idxs, _ = acquisition_scores.sample_proportionally(
                probability_masses, coreset_size
            )
        elif self.proposal == "softmax":
            idxs, _ = acquisition_scores.sample_softmax(probability_masses, coreset_size, self.temperature)
        return idxs, torch.ones(coreset_size)