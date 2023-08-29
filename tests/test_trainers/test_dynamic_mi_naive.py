import torch
import unittest

from mutual_information.configs.dynamic_mi_naive_config import DynamicMutualInformationNaiveConfig
from mutual_information.data.dataloaders import CorrelationCoefficientGaussianLoaderConfig

from mutual_information.trainers.mi_trainer import DynamicMutualInformationTrainerNaive
class TestDynamicMINaive(unittest.TestCase):

    def setUp(self) -> None:
        self.config = DynamicMutualInformationNaiveConfig()
        self.config.experiment_indentifier = "dmin_trainer_unittest2"
        self.config.dataloader.number_of_time_steps = 5
        self.config.trainer.number_of_epochs = 1000


    def test_trainer(self):
        DMIT = DynamicMutualInformationTrainerNaive(self.config)
        DMIT.train()
        self.assertIsNotNone(DMIT)

        #DMIT.train()


if __name__=="__main__":
    print()