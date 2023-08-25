
import torch
from torch.optim import Adam

import numpy as np
from pprint import pprint
from dataclasses import asdict

from mutual_information.models.binary_classifiers import BaseBinaryClassifier
from mutual_information.data.dataloaders import ContrastiveMultivariateGaussianLoader
from mutual_information.models.binary_classifier_config import BaseBinaryClassifierConfig
from mutual_information.configs.mi_config import MutualInformationConfig
from mutual_information.losses.contrastive_loss import contrastive_loss

class MutualInformationTrainer:

    name_="mutual_information_estimator"

    def __init__(self,
                 config: MutualInformationConfig,
                 contrastive_dataloader:ContrastiveMultivariateGaussianLoader,
                 binary_classifier:BaseBinaryClassifier):

        self.config = config
        self.learning_rate = config.trainer.learning_rate
        self.number_of_epochs = config.trainer.number_of_epochs
        self.device = torch.device(config.trainer.device)

        self.contrastive_dataloader = contrastive_dataloader
        self.constrastive_loss = contrastive_loss

        self.binary_classifier = binary_classifier
        self.binary_classifier.to(self.device)

        self.contrastive_dataloader = contrastive_dataloader

    def parameters_info(self):
        print("# ==================================================")
        print("# START OF BACKWARD MI TRAINING ")
        print("# ==================================================")
        print("# Binary Classifier *********************************")
        pprint(asdict(self.binary_classifier.config))
        print("# Paths Parameters **********************************")
        pprint(asdict(self.contrastive_dataloader.config))
        print("# Trainer Parameters")
        pprint(asdict(self.config))
        print("# ==================================================")
        print("# Number of Epochs {0}".format(self.number_of_epochs))
        print("# ==================================================")

    def preprocess_data(self,data_batch):
        data_batch["join"] = data_batch["join"].to(self.device)
        data_batch["independent"] = data_batch["independent"].to(self.device)
        return data_batch

    def train_step(self,data_batch,number_of_training_step):
        data_batch = self.preprocess_data(data_batch)
        loss = self.constrastive_loss(data_batch,self.binary_classifier)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar('training loss', loss, number_of_training_step)
        return loss

    def test_step(self,data_batch):
        with torch.no_grad():
            data_batch = self.preprocess_data(data_batch)
            loss_ = self.constrastive_loss(data_batch, self.binary_classifier)
            return loss_

    def initialize(self):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.config.experiment_files.tensorboard_path)

        self.optimizer = Adam(self.binary_classifier.parameters(),lr=self.learning_rate)
        data_batch = next(self.contrastive_dataloader.train().__iter__())
        data_batch = self.preprocess_data(data_batch)
        initial_loss = self.constrastive_loss(data_batch,self.binary_classifier)

        assert torch.isnan(initial_loss).any() == False
        assert torch.isinf(initial_loss).any() == False

        self.save_results(self.binary_classifier,
                          initial_loss,
                          None,
                          None,
                          None,
                          0,
                          checkpoint=True)

        return initial_loss

    def train(self):
        initial_loss = self.initialize()
        best_loss = initial_loss

        number_of_training_step = 0
        number_of_test_step = 0
        for epoch in range(self.number_of_epochs):

            LOSS = []
            train_loss = []
            for data_batch in self.contrastive_dataloader.train():
                loss = self.train_step(data_batch,number_of_training_step)
                train_loss.append(loss.item())
                LOSS.append(loss.item())
                number_of_training_step += 1
            average_train_loss = np.asarray(train_loss).mean()

            test_loss = []
            for data_batch in self.contrastive_dataloader.test():
                loss = self.test_step(data_batch)
                test_loss.append(loss.item())
                number_of_test_step+=1
            average_test_loss = np.asarray(test_loss).mean()

            # SAVE RESULTS IF LOSS DECREASES IN VALIDATION
            if average_test_loss < best_loss:
                self.save_results(self.binary_classifier,
                                  initial_loss,
                                  average_train_loss,
                                  average_test_loss,
                                  LOSS,
                                  epoch,
                                  checkpoint=False)

            if (epoch + 1) % self.config.trainer.save_model_epochs == 0:
                self.save_results(self.binary_classifier,
                                  initial_loss,
                                  average_train_loss,
                                  average_test_loss,
                                  LOSS,
                                  epoch+1,
                                  checkpoint=True)

            if epoch % 10 == 0:
                print("Epoch: {}, Loss: {}".format(epoch + 1, average_train_loss))

        self.writer.close()

    def save_results(self,
                     binary_classifier,
                     initial_loss,
                     average_train_loss,
                     average_test_loss,
                     LOSS,
                     epoch=0,
                     checkpoint=False):
        if checkpoint:
            RESULTS = {
                "binary_classifier":binary_classifier,
                "initial_loss":initial_loss,
                "average_train_loss":average_train_loss,
                "average_test_loss":average_test_loss,
                "LOSS":LOSS
            }
            torch.save(RESULTS,self.config.experiment_files.best_model_path_checkpoint.format(epoch))
        else:
            RESULTS = {
                "binary_classifier":binary_classifier,
                "initial_loss":initial_loss,
                "average_train_loss":average_train_loss,
                "average_test_loss":average_test_loss,
                "LOSS":LOSS
            }
            torch.save(RESULTS,self.config.experiment_files.best_model_path)


