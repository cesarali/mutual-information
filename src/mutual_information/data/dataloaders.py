import torch
from pprint import pprint
from torch import matmul as m
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from abc import ABC
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split
from typing import Union,Tuple,List

class BasicDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DictDataSet(Dataset):
    """
    # Define your data dictionary
    data_dict = {'input': torch.randn(2, 10), 'target': torch.randn(2, 5)}

    # Create your dataset
    my_dataset = DictDataSet(data_dict)

    # Create a DataLoader from your dataset
    batch_size = 2
    dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
    """
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())

    def __len__(self):
        return len(self.data_dict[self.keys[0]])

    def __getitem__(self, idx):
        return {key: self.data_dict[key][idx] for key in self.keys}

class BaseDataLoader(ABC):
    name_="base_data_loader"
    def __init__(self,**kwargs):
        super(BaseDataLoader,self).__init__()

    def define_dataset_and_dataloaders(self,X,training_proportion=0.8,batch_size=32):
        self.batch_size = batch_size
        if isinstance(X,torch.Tensor):
            dataset = TensorDataset(X)
        elif isinstance(X,dict):
            dataset = DictDataSet(X)

        self.total_data_size = len(dataset)
        self.training_data_size = int(training_proportion * self.total_data_size)
        self.test_data_size = self.total_data_size - self.training_data_size

        training_dataset, test_dataset = random_split(dataset, [self.training_data_size, self.test_data_size])
        self._train_iter = DataLoader(training_dataset, batch_size=batch_size)
        self._test_iter = DataLoader(test_dataset, batch_size=batch_size)

    def train(self):
        return self._train_iter

    def test(self):
        return self._test_iter


from dataclasses import dataclass

@dataclass
class ContrastiveMultivariateGaussianLoaderConfig:
    name:str = "ContrastiveMultivariateGaussianLoader"
    dimensions_per_variable: int = 2
    number_of_variables: int = 2
    batch_size: int = 32
    sample_size: int = 300

class ContrastiveMultivariateGaussianLoader(BaseDataLoader):
    """
    """
    name_ = "contrastive_multivariate_gaussian"

    def __init__(self, config:ContrastiveMultivariateGaussianLoaderConfig):
        self.config = config
        self.dimensions_per_variable = config.dimensions_per_variable
        self.number_of_variables = config.number_of_variables
        self.total_dimensions = self.dimensions_per_variable * self.number_of_variables
        self.sample_size = config.sample_size
        self.batch_size = config.batch_size

        sample_join,sample_indendent = self.sample()
        sample = {"join":sample_join,"independent":sample_indendent}
        self.define_dataset_and_dataloaders(sample,batch_size=self.batch_size)

    def obtain_parameters(self):
        return self.parameters_

    def mutual_information(self):
        det_1 = torch.det(self.covariance_1)
        det_2 = torch.det(self.covariance_2)
        det_full = torch.det(self.covariance_full)

        entropy_1 = .5 * torch.log(det_1)
        entropy_2 = .5 * torch.log(det_2)
        entropy_full = .5 * torch.log(det_full)

        mutual_information = entropy_1 + entropy_2 - entropy_full
        return mutual_information

    def sample(self):
        self.mean_1 = torch.zeros(self.dimensions_per_variable)
        self.mean_2 = torch.zeros(self.dimensions_per_variable)
        self.mean_full = torch.zeros(self.total_dimensions)

        covariance_full = torch.Tensor(size=(self.total_dimensions,
                                             self.total_dimensions)).normal_(0., 1.)
        self.covariance_full = m(covariance_full,covariance_full.T)

        self.covariance_1 = self.covariance_full[:self.dimensions_per_variable,:self.dimensions_per_variable]
        self.covariance_2 = self.covariance_full[self.dimensions_per_variable:,self.dimensions_per_variable:]

        normal_1 = MultivariateNormal(self.mean_1, self.covariance_1)
        normal_2 = MultivariateNormal(self.mean_2, self.covariance_2)
        normal_full = MultivariateNormal(self.mean_full, self.covariance_full)

        sample_1 = normal_1.sample(sample_shape=(self.sample_size,))
        sample_2 = normal_2.sample(sample_shape=(self.sample_size,))
        sample_join = normal_full.sample(sample_shape=(self.sample_size,))
        sample_indendent = torch.cat([sample_1, sample_2], dim=1)

        return sample_join,sample_indendent

