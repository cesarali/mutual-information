import torch
import unittest
from pprint import pprint
from dataclasses import asdict
from mutual_information.data.dataloader_utils import load_dataloader
from mutual_information.configs.mi_config import get_config_from_file
from mutual_information.configs.mi_config import MutualInformationConfig
from mutual_information.data.dataloaders import ContrastiveMultivariateGaussianLoaderConfig


class TestMIDataloader(unittest.TestCase):

    read_config = MutualInformationConfig

    def setUp(self):
        self.batch_size = 23
        self.number_of_variables = 2
        self.dimensions_per_variable = 1
        self.expected_size = torch.Size([self.batch_size,self.dimensions_per_variable*self.number_of_variables])

        self.config = MutualInformationConfig(experiment_name='mi',
                                              experiment_type='multivariate_gaussian',
                                              experiment_indentifier="mi_unittest",
                                              delete=True)
        self.config.dataloader = ContrastiveMultivariateGaussianLoaderConfig(dimensions_per_variable=self.dimensions_per_variable,
                                                                             number_of_variables=self.number_of_variables,
                                                                             sample_size=1000,
                                                                             batch_size=self.batch_size,
                                                                             data_set="example_basic",
                                                                             delete_data=True)

    def test_dataloader(self):
        dataloader = load_dataloader(self.config)
        databatch = next(dataloader.train().__iter__())
        print("Join")
        print(databatch['join'].shape)
        print("Independent")
        print(databatch['independent'].shape)
        self.assertIsNotNone(dataloader)
        self.assertEqual(self.expected_size,databatch['join'].shape)
        self.assertEqual(self.expected_size,databatch['independent'].shape)

        #for databatch in dataloader.train():
        #    print(databatch["join"].shape)

    def test_mi(self):
        dataloader = load_dataloader(self.config)
        MI = dataloader.mutual_information()
        print(MI)
        self.assertFalse(torch.isnan(MI).all())



if __name__=="__main__":
    unittest.main()
