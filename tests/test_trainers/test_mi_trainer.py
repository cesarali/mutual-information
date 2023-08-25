import torch
import unittest
from pprint import pprint
from dataclasses import asdict

import torch
import unittest
from pprint import pprint
from dataclasses import asdict
from mutual_information.data.dataloader_utils import load_dataloader
from mutual_information.configs.mi_config import get_config_from_file
from mutual_information.configs.mi_config import MutualInformationConfig
from mutual_information.data.dataloaders import ContrastiveMultivariateGaussianLoaderConfig
from mutual_information.trainers.mi_trainer_config import MITrainerConfig
from mutual_information.models.binary_classifier_config import BaseBinaryClassifierConfig
from mutual_information.models.models_utils import load_binary_classifier
from mutual_information.trainers.mi_trainer import MutualInformationTrainer
from mutual_information.trainers.trainers_utils import load_experiments_configuration,load_experiments_results

class TestMITrainer(unittest.TestCase):

    read_config = MutualInformationConfig

    def setUp(self):
        self.config = MutualInformationConfig(experiment_name='mi',
                                              experiment_type='multivariate_gaussian',
                                              experiment_indentifier="mi_unittest2",
                                              delete=True)
        self.config.trainer = MITrainerConfig(number_of_epochs=50,
                                              save_model_epochs=10)

        self.config.dataloader = ContrastiveMultivariateGaussianLoaderConfig(sample_size=1000,
                                                                             batch_size=32)
        self.config.binary_classifier = BaseBinaryClassifierConfig(hidden_size=40)

        self.config.initialize_new_experiment()
        self.contrastive_dataloader = load_dataloader(self.config)
        self.binary_classifier = load_binary_classifier(self.config)

    def test_setup(self):
        self.read_config = get_config_from_file(experiment_name='mi',
                                                experiment_type='multivariate_gaussian',
                                                experiment_indentifier="mi_unittest2")
        pprint(asdict(self.read_config.dataloader))
        pprint(asdict(self.read_config.binary_classifier))
        pprint(asdict(self.read_config.trainer))

    def test_binary_classifier(self):
        databatch  = next(self.contrastive_dataloader.train().__iter__())
        x_join = databatch["join"]
        binary_forward = self.binary_classifier(x_join)
        print(binary_forward.shape)

    def test_trainer_setup(self):
        MIT = MutualInformationTrainer(self.config,
                                       self.contrastive_dataloader,
                                       self.binary_classifier)
        #databatch = next(self.contrastive_dataloader.train().__iter__())
        #databatch = MIT.preprocess_data(databatch)
        MIT.train()

        #results = load_experiments_results(experiment_name='mi',
        #                                   experiment_type='multivariate_gaussian',
        #                                   experiment_indentifier="mi_unittest2")

    def test_load_experiments(self):
        binary_classifier, dataloader = load_experiments_configuration(experiment_name='mi',
                                                                       experiment_type='multivariate_gaussian',
                                                                       experiment_indentifier="mi_unittest2")
        databatch = next(dataloader.train().__iter__())
        x_join = databatch["join"]
        forward_classifier = binary_classifier(x_join)
        print(forward_classifier.shape)


if __name__=="__main__":
    unittest.main()
