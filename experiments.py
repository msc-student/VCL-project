import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from trainer import Trainer
from acquisition_scores import entropy_score, mutual_information
import coreset
import copy

class Permute(object):
    def __init__(self, permutation):
        self.permutation = permutation

    def __call__(self, sample):
        return sample[self.permutation]


class MultiTaskDataset:
    def __init__(self, dataset, task):
        self.dataset = dataset
        self.task = task


    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y, self.task

    def __len__(self):
        return len(self.dataset)


class PermutedSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def raw_data(self, index):
        return self.subset[index][0]
        
    def __len__(self):
        return len(self.subset)

class PermutedMNIST:
    def __init__(self, n_tasks, input_dim):
        self.n_tasks = n_tasks
        self.input_dim = input_dim
        self.random_perm = [torch.randperm(input_dim) for _ in range(n_tasks)]
        self.dataset_train = [datasets.MNIST('./mnist', 
                                             train=True, 
                                             download=False, 
                                             transform=self.get_transform(perm)) for perm in self.random_perm]
        self.dataset_test = [datasets.MNIST('./mnist', 
                                             train=False, 
                                             download=False, 
                                             transform=self.get_transform(perm)) for perm in self.random_perm]

    def get_transform(self, permutation):
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Lambda(lambda x: torch.flatten(x)),
                                   Permute(permutation)])
        
    def __len__(self):
        return self.n_tasks

class PermutedFashionMNIST:
    def __init__(self, n_tasks, input_dim):
        self.n_tasks = n_tasks
        self.input_dim = input_dim
        self.random_perm = [torch.randperm(input_dim) for _ in range(n_tasks)]
        self.dataset_train = [datasets.FashionMNIST('./mnist', 
                                             train=True, 
                                             download=True, 
                                             transform=self.get_transform(perm)) for perm in self.random_perm]
        self.dataset_test = [datasets.FashionMNIST('./mnist', 
                                             train=False, 
                                             download=True, 
                                             transform=self.get_transform(perm)) for perm in self.random_perm]

    def get_transform(self, permutation):
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Lambda(lambda x: torch.flatten(x)),
                                   Permute(permutation)])
        
    def __len__(self):
        return self.n_tasks

class SplitDigitSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset, initial_digit):
        self.mnist_dataset = mnist_dataset
        self.initial_digit = torch.tensor(initial_digit)
        self.indices = torch.argwhere((mnist_dataset.targets == initial_digit) | (mnist_dataset.targets == initial_digit+1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        mnist_index = self.indices[index].item()
        return self.mnist_dataset[mnist_index][0], (self.mnist_dataset[mnist_index][1] == self.initial_digit+1).long()

class SplitMNIST:
    def __init__(self, n_tasks):
        self.n_tasks = n_tasks
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: torch.flatten(x)),
                    ])
        self.mnist_dataset_train = datasets.MNIST('./mnist', train=True, download=True, transform=transform)
        self.mnist_dataset_test = datasets.MNIST('./mnist', train=False, download=True, transform=transform)
        self.dataset_train = [SplitDigitSubsetDataset(self.mnist_dataset_train, 2*i) for i in range(self.n_tasks)]
        self.dataset_test = [SplitDigitSubsetDataset(self.mnist_dataset_test, 2*i) for i in range(self.n_tasks)]
    
    def __len__(self):
        return self.n_tasks

    def __getitem__(self, index):
        mnist_index = self.indices[index]
        return self.mnist_dataset[mnist_index]

class SplitFashionMNIST:
    def __init__(self, n_tasks):
        self.n_tasks = n_tasks
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: torch.flatten(x)),
                    ])
        self.mnist_dataset_train = datasets.FashionMNIST('./mnist', train=True, download=True, transform=transform)
        self.mnist_dataset_test = datasets.FashionMNIST('./mnist', train=False, download=True, transform=transform)
        self.dataset_train = [SplitDigitSubsetDataset(self.mnist_dataset_train, 2*i) for i in range(self.n_tasks)]
        self.dataset_test = [SplitDigitSubsetDataset(self.mnist_dataset_test, 2*i) for i in range(self.n_tasks)]
    
    def __len__(self):
        return self.n_tasks

    def __getitem__(self, index):
        mnist_index = self.indices[index]
        return self.mnist_dataset[mnist_index]


class Experiment:
    def __init__(self,
                model,
                n_tasks,
                heads,
                datasets,
                lr,
                batch_size,
                n_epochs,
                device,
                coreset_method=None,
                coreset_size=0,
                importance_sampler=coreset.ImportanceSampling("proportional", 
                                                             entropy_score, 
                                                             50, 
                                                             10,
                                                             temperature=1.),
                acquisition_function=mutual_information,
                verbose=True):
        self.model = model
        self.n_tasks = n_tasks
        self.heads = heads
        self.datasets = datasets
        self.verbose = verbose
        self.device = device
        self.trainer = Trainer(
                              lr=lr,
                              batch_size=batch_size,
                              n_epochs=n_epochs,
                              device=device,
                              optimizer=torch.optim.Adam)
        self.coreset = coreset.Coreset(coreset_method, coreset_size, n_epochs, device, importance_sampler)
        print(f"Running experiment - With {coreset_method} coreset method - With {coreset_size} samples")

    
    def run(self, save_models=False):
        accuracies = torch.zeros((self.n_tasks, self.n_tasks))
        saved_models = []
        for i in range(self.n_tasks):
            dataset_train = self.coreset.get_indices(self.datasets.dataset_train[i])
            train_loader = DataLoader(dataset=dataset_train,
                                      batch_size=self.trainer.batch_size,
                                      shuffle=True)
            print(f'Training task {i} - {len(dataset_train)} samples')
            self.trainer.training_task(self.model, train_loader, head=self.heads[i])
            # finetuning a copy of the model
            if (self.coreset.coreset_method is None):
                new_model = self.model
            else:
                new_model = self.coreset.train(initial_model=self.model, 
                                                train_datasets=self.coreset.coresets, 
                                                heads=self.heads[:i+1])
            for j in range(i+1):
                accuracies[i,j] = self.test(new_model,
                                            self.datasets.dataset_test[j], 
                                            self.heads[j])
                if self.verbose:
                    print(f'Step {i} - Task {j} - Accuracy {accuracies[i, j]}')
            if self.verbose:
                print(f'Average accuracy {accuracies[i,:i+1].mean()}')
            if save_models:
                saved_models.append(copy.deepcopy(self.model))
        self.saved_models = saved_models
        return accuracies

    def finetuning(self, models, importance_sampler=None):
        # Replay Variational Continual Learning
        # coreset selection + retraining
        accuracies = torch.zeros((self.n_tasks, self.n_tasks))
        if importance_sampler:
            self.coreset.importance_sampler = importance_sampler
        for i in range(self.n_tasks):
            self.coreset.coresets = []
            for j in range(i+1):
                self.coreset.get_indices(self.datasets.dataset_train[j],
                                        model=models[j],
                                        head=self.heads[j])
            if self.coreset.coreset_method == 'importance':
                new_model = self.coreset.train_importance(initial_model=models[i], 
                                                        train_datasets=self.coreset.coresets, 
                                                        heads=self.heads[:i+1])
            elif self.coreset.coreset_method == 'random':
                new_model = self.coreset.train(initial_model=models[i],
                                               train_datasets=self.coreset.coresets,
                                               heads=self.heads[:i+1])
            else:
                raise ValueError
            for j in range(i+1):
                accuracies[i, j] = self.test(new_model,
                                             self.datasets.dataset_test[j],
                                             self.heads[j])
        return accuracies


    def test(self, model, test_dataset, head):
        model.eval()
        test_loader = DataLoader(test_dataset, self.trainer.batch_size, shuffle=False)
        with torch.no_grad():
            predictions = []
            targets = []
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                predictions.append(model.prediction(x, head=head))
                targets.append(y)
            predictions = torch.concat(predictions)
            targets = torch.concat(targets)
            predictions = torch.argmax(predictions, dim=1)
        return (predictions.cpu() == targets.cpu()).sum()/len(targets)
