from typing import Union

from mutual_information.configs.mi_config import MutualInformationConfig
from mutual_information.configs.dynamic_mi_naive_config import DynamicMutualInformationNaiveConfig

from mutual_information.data.dataloaders import ContrastiveMultivariateGaussianLoader
from mutual_information.data.dataloaders import CorrelationCoefficientGaussianLoader

from mutual_information.data.dataloaders import ContrastiveMultivariateGaussianLoaderConfig


def load_dataloader(config:Union[MutualInformationConfig,ContrastiveMultivariateGaussianLoaderConfig,DynamicMutualInformationNaiveConfig]):
    if isinstance(config,MutualInformationConfig):
        config_ = config.dataloader
    elif isinstance(config,DynamicMutualInformationNaiveConfig):
        config_ = config.dataloader
    elif isinstance(config, ContrastiveMultivariateGaussianLoaderConfig):
        config_ = config
    else:
        raise Exception("Config Does Not Exist")

    if config_.name == "ContrastiveMultivariateGaussianLoader":
        dataloader = ContrastiveMultivariateGaussianLoader(config_)
    elif config_.name == "CorrelationCoefficientGaussianLoader":
        dataloader = CorrelationCoefficientGaussianLoader(config_)
    else:
        raise Exception("Dataloader Does Not Exist")

    return dataloader

