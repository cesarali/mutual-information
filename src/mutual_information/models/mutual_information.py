import os
import sys
import torch
from dataclasses import dataclass, asdict
EPSILON = 1e-12


# load models
from mutual_information.data.dataloader_utils import load_dataloader
from mutual_information.models.models_utils import load_binary_classifier
from mutual_information.trainers.trainers_utils import load_experiments_configuration

# configs
from mutual_information.configs.mi_config import MutualInformationConfig

# models
from mutual_information.models.binary_classifiers import BaseBinaryClassifier
from mutual_information.data.dataloaders import ContrastiveMultivariateGaussianLoader

@dataclass
class MutualInformationEstimator:
    """

    """
    binary_classifier: BaseBinaryClassifier = None
    dataloader: ContrastiveMultivariateGaussianLoader = None

    def create_new_from_config(self, config:MutualInformationConfig, device=torch.device("cpu")):
        self.config = config
        self.config.initialize_new_experiment()

        self.dataloader = load_dataloader(self.config)
        self.binary_classifier = load_binary_classifier(self.config)
        self.binary_classifier.to(device)

    def load_results_from_directory(self,
                                    experiment_name='mi',
                                    experiment_type='multivariate_gaussian',
                                    experiment_indentifier="test",
                                    checkpoint=None,
                                    device=torch.device("cpu")):
        self.binary_classifier, self.dataloader = load_experiments_configuration(experiment_name,
                                                                                 experiment_type,
                                                                                 experiment_indentifier,
                                                                                 checkpoint)
        self.binary_classifier.to(device)

    def MI_Estimate(self):
        log_q = 0.
        number_of_pairs = 0
        for databath in self.dataloader.train():
            #select data
            x_join = databath["join"]

            #calculate probability
            q = self.binary_classifier(x_join)
            assert torch.isnan(q).any() == False
            assert torch.isinf(q).any() == False

            #average
            log_q = torch.log(q + EPSILON)
            log_q = log_q.sum()
            number_of_pairs += x_join.shape[0]

        log_q_av = log_q/number_of_pairs
        return log_q_av

