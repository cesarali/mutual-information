import torch
import unittest
import numpy as np

from mutual_information.configs.dynamic_mi_naive_config import DynamicMutualInformationNaiveConfig
from mutual_information.data.dataloader_utils import load_dataloader
from mutual_information.data.dataloaders import ContrastiveMultivariateGaussianLoaderConfig

from pprint import pprint
from dataclasses import dataclass, asdict

class TestCorrelatedGaussianLoader(unittest.TestCase):

    def test_ccg_dataloader(self):
        config = DynamicMutualInformationNaiveConfig()
        config.dataloader.sample_size = 2000
        config.dataloader.delete_data = True
        config.dataloader.data_set = "dataloader_unittest"

        device = torch.device(config.trainer.device)
        dataloader = load_dataloader(config)
        databatch = next(dataloader.train().__iter__())

        data_join = []
        data_independent = []
        for databatch in dataloader.train():
            data_join.append(databatch["join"])
            data_independent.append(databatch["independent"])
        data_join = torch.vstack(data_join)
        data_independent = torch.vstack(data_independent)

        for time_step in range(1,dataloader.number_of_time_steps):
            rho_from_sample = np.corrcoef(data_join[:, 0], data_join[:, time_step])[0, 1]
            rho_from_sample_ind = np.corrcoef(data_independent[:, 0], data_independent[:, time_step])[0, 1]

            rho_from_parameters = dataloader.rho_values[time_step].item()
            self.assertAlmostEqual(rho_from_sample,rho_from_parameters,delta=0.1)
            self.assertAlmostEqual(rho_from_sample_ind,0.,delta=0.1)

if __name__=="__main__":
    unittest.main()