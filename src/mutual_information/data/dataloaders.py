import os
import torch

from pathlib import Path
from torch import matmul as m
from torch.distributions import MultivariateNormal
from torch.distributions import Normal


from abc import ABC
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split
from mutual_information.utils.covariance_functions import copy_upper_diagonal_values

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
    data_set:str = "example"
    dataloader_data_dir:str = None

    dimensions_per_variable: int = 2
    number_of_variables: int = 2
    batch_size: int = 32
    sample_size: int = 300
    delete_data:bool = False

    def __post_init__(self):
        from mutual_information import data_path
        self.dataloader_data_dir = os.path.join(data_path,"raw",self.name)
        self.dataloader_data_dir_file = os.path.join(self.dataloader_data_dir,self.data_set+".tr")


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
        self.delete_data = config.delete_data

        self.dataloader_data_dir = config.dataloader_data_dir
        self.dataloader_data_dir_path = Path(self.dataloader_data_dir)
        self.dataloader_data_dir_file_path = Path(config.dataloader_data_dir_file)

        sample = self.obtain_sample()
        self.define_dataset_and_dataloaders(sample,batch_size=self.batch_size)

    def set_parameters(self,parameters):
        self.mean_1 = parameters["mean_1"]
        self.mean_2 = parameters["mean_2"]
        self.mean_full = parameters["mean_full"]
        self.covariance_1 = parameters["covariance_1"]
        self.covariance_2 = parameters["covariance_2"]
        self.covariance_full = parameters["covariance_full"]


    def obtain_sample(self):
        if self.dataloader_data_dir_file_path.exists():
            if self.delete_data:
                parameters, sample_join, sample_indendent = self.sample()
                sample = {"join": sample_join, "independent": sample_indendent}
                torch.save({"sample":sample,"parameters":parameters},
                           self.dataloader_data_dir_file_path)
            else:
                parameters_and_data = torch.load(self.dataloader_data_dir_file_path)
                sample = parameters_and_data["sample"]
                parameters = parameters_and_data["parameters"]
                self.set_parameters(parameters)
        else:
            if not self.dataloader_data_dir_path.exists():
                os.makedirs(self.dataloader_data_dir_path)
            parameters, sample_join, sample_indendent = self.sample()
            sample = {"join": sample_join,
                      "independent": sample_indendent}
            torch.save({"sample":sample,"parameters":parameters}, self.dataloader_data_dir_file_path)
        return sample

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
                                             self.total_dimensions)).normal_(10., 2.)

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

        distributions_parameters = {"mean_1":self.mean_1,
                                    "mean_2":self.mean_2,
                                    "mean_full":self.mean_full,
                                    "covariance_1":self.covariance_1,
                                    "covariance_2":self.covariance_2,
                                    "covariance_full":self.covariance_full}

        return distributions_parameters,sample_join,sample_indendent

@dataclass
class CorrelationCoefficientGaussianLoaderConfig:
    name:str = "CorrelationCoefficientGaussianLoader"
    data_set:str = "correlation_timeseries_example"
    dataloader_data_dir:str = None

    number_of_time_steps: int = 10
    batch_size: int = 32
    sample_size: int = 2000
    delete_data:bool = False

    def __post_init__(self):
        from mutual_information import data_path
        self.dataloader_data_dir = os.path.join(data_path,"raw",self.name)
        self.dataloader_data_dir_file = os.path.join(self.dataloader_data_dir,self.data_set+".tr")

class CorrelationCoefficientGaussianLoader(BaseDataLoader):
    """

    """
    def __init__(self,config:CorrelationCoefficientGaussianLoaderConfig):
        self.config = config
        self.sample_size = config.sample_size
        self.batch_size = config.batch_size
        self.delete_data = config.delete_data
        self.number_of_time_steps = config.number_of_time_steps

        self.dataloader_data_dir = config.dataloader_data_dir
        self.dataloader_data_dir_path = Path(self.dataloader_data_dir)
        self.dataloader_data_dir_file_path = Path(config.dataloader_data_dir_file)

        # Set the mean and covariance matrix
        self.mu = torch.zeros(self.number_of_time_steps)
        sample = self.obtain_sample()
        self.define_dataset_and_dataloaders(sample,batch_size=self.batch_size)

    def sample(self):
        all_rho_values = self.obtain_rolling_correlations(self.number_of_time_steps)
        covariance = copy_upper_diagonal_values(all_rho_values)
        independent_covariace = torch.diag(covariance)

        mu = torch.zeros(self.number_of_time_steps)
        join_distribution = MultivariateNormal(mu, covariance)
        independent_distribution = Normal(mu,independent_covariace)

        join_sample = join_distribution.sample((self.sample_size,))
        independent_sample = independent_distribution.sample((self.sample_size,))
        parameters = {"rho_values":self.rho_values}

        return parameters,join_sample,independent_sample

    def set_parameters(self,parameters):
        self.rho_values = parameters["rho_values"]

    def obtain_sample(self):
        if self.dataloader_data_dir_file_path.exists():
            if self.delete_data:
                parameters, sample_join, sample_indendent = self.sample()
                sample = {"join": sample_join, "independent": sample_indendent}
                torch.save({"sample":sample,"parameters":parameters},
                           self.dataloader_data_dir_file_path)
            else:
                parameters_and_data = torch.load(self.dataloader_data_dir_file_path)
                sample = parameters_and_data["sample"]
                parameters = parameters_and_data["parameters"]
                self.set_parameters(parameters)
        else:
            if not self.dataloader_data_dir_path.exists():
                os.makedirs(self.dataloader_data_dir_path)
            parameters, sample_join, sample_indendent = self.sample()
            sample = {"join": sample_join,
                      "independent": sample_indendent}
            torch.save({"sample":sample,"parameters":parameters}, self.dataloader_data_dir_file_path)
        return sample

    def obtain_rolling_correlations(self,number_of_time_steps=4):
        self.rho_values = torch.linspace(0., 1., number_of_time_steps)
        self.rho_values = torch.flip(self.rho_values, dims=(0,))
        all_rho_values = []
        for time_step in range(number_of_time_steps):
            column = torch.zeros(number_of_time_steps)
            rolled_rhos = torch.roll(self.rho_values, shifts=time_step)
            column[time_step:] = rolled_rhos[time_step:]
            all_rho_values.append(column)
        all_rho_values = torch.vstack(all_rho_values)
        return all_rho_values

    def mutual_information(self):
        return -.5*torch.log(1.-(self.rho_values)**2.)

