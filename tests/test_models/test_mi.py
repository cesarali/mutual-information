import torch
import unittest
from pprint import pprint
from dataclasses import asdict

import torch
import unittest
from pprint import pprint
from dataclasses import asdict
from mutual_information.configs.mi_config import get_config_from_file
from mutual_information.configs.mi_config import MutualInformationConfig
from mutual_information.data.dataloaders import ContrastiveMultivariateGaussianLoaderConfig
from mutual_information.trainers.mi_trainer_config import MITrainerConfig
from mutual_information.models.binary_classifier_config import BaseBinaryClassifierConfig

from mutual_information.data.dataloader_utils import load_dataloader
from mutual_information.models.models_utils import load_binary_classifier

from mutual_information.trainers.mi_trainer import MutualInformationTrainer
from mutual_information.trainers.trainers_utils import load_experiments_configuration
from mutual_information.trainers.trainers_utils import load_experiments_results
from mutual_information.models.mutual_information import MutualInformationEstimator

class TestMITrainer(unittest.TestCase):

    read_config = MutualInformationConfig

    def setUp(self):
        self.config = MutualInformationConfig(experiment_name='mi',
                                              experiment_type='multivariate_gaussian',
                                              experiment_indentifier="mi_unittest",
                                              delete=True)
        self.config.dataloader = ContrastiveMultivariateGaussianLoaderConfig(sample_size=1000,
                                                                             batch_size=32,
                                                                             data_set="example_big",
                                                                             delete_data=False)
        self.config.binary_classifier = BaseBinaryClassifierConfig(hidden_size=40)
        self.config.trainer = MITrainerConfig(number_of_epochs=5,
                                              save_model_epochs=2)
        self.MI = MutualInformationEstimator()


    def test_trainer_setup(self):
        self.MI.create_new_from_config(self.config)

#    @unittest.skip
    def test_load(self):
        MIE = MutualInformationEstimator()
        MIE.load_results_from_directory(experiment_name='mi',
                                        experiment_type='multivariate_gaussian',
                                        experiment_indentifier="mi_trainer_big",
                                        checkpoint=None)
        databath = next(MIE.dataloader.train().__iter__())
        x_join = databath["join"]
        x_independent = databath["independent"]
        p_join = MIE.binary_classifier(x_join)
        p_independent = MIE.binary_classifier(x_independent)

        print("p join: {0} p independent: {1}".format(p_join.mean(),p_independent.mean()))

        estimate = MIE.MI_Estimate()
        real_value = MIE.dataloader.mutual_information()

        print("estimate: {0} real: {1}".format(estimate.item(),real_value.item()))


if __name__=="__main__":
    unittest.main()

