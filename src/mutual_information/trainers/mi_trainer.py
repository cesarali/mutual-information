import torch
from torch.optim import Adam

import numpy as np
from pprint import pprint
from dataclasses import asdict

from mutual_information.models.binary_classifiers import BaseBinaryClassifier
from mutual_information.data.dataloaders import ContrastiveMultivariateGaussianLoader
from mutual_information.data.dataloaders import CorrelationCoefficientGaussianLoader
from mutual_information.models.binary_classifier_config import BaseBinaryClassifierConfig
from mutual_information.configs.mi_config import MutualInformationConfig
from mutual_information.configs.dynamic_mi_naive_config import DynamicMutualInformationNaiveConfig
from mutual_information.losses.contrastive_loss import contrastive_loss
from mutual_information.losses.mine import mine_loss
from mutual_information.models.mutual_information import MutualInformationEstimator
from mutual_information.models.dynamic_mutual_information import DynamicMutualInformationNaiveEstimator

class MutualInformationTrainer:

    name_="mutual_information_estimator"

    def __init__(self,
                 config: MutualInformationConfig,
                 contrastive_dataloader:ContrastiveMultivariateGaussianLoader=None,
                 binary_classifier:BaseBinaryClassifier=None):

        self.config = config
        self.learning_rate = config.trainer.learning_rate
        self.number_of_epochs = config.trainer.number_of_epochs
        self.device = torch.device(config.trainer.device)
        self.loss_type = config.trainer.loss_type

        if self.loss_type == "mine":
            self.loss = mine_loss
        elif self.loss_type == "contrastive":
            self.loss = contrastive_loss

        if binary_classifier is not None:
            self.contrastive_dataloader = contrastive_dataloader
            self.binary_classifier = binary_classifier
            self.binary_classifier.to(self.device)
        else:
            MIE = MutualInformationEstimator()
            MIE.create_new_from_config(self.config,self.device)
            self.contrastive_dataloader = MIE.dataloader
            self.binary_classifier = MIE.binary_classifier

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
        loss = self.loss(data_batch,self.binary_classifier)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar('training loss', loss, number_of_training_step)
        return loss

    def test_step(self,data_batch):
        with torch.no_grad():
            data_batch = self.preprocess_data(data_batch)
            loss_ = self.loss(data_batch, self.binary_classifier)
            return loss_

    def initialize(self):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.config.experiment_files.tensorboard_path)

        self.optimizer = Adam(self.binary_classifier.parameters(),lr=self.learning_rate)
        data_batch = next(self.contrastive_dataloader.train().__iter__())
        data_batch = self.preprocess_data(data_batch)
        initial_loss = self.loss(data_batch,self.binary_classifier)

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

class DynamicMutualInformationTrainerNaive:
    """
    This is mainly a test for the 2 variables estimator (one neural netowrk per step)
    to obtain the MINE graph with the gaussian correlation variables

    """
    name_="mutual_information_estimator"

    def __init__(self,
                 config: DynamicMutualInformationNaiveConfig,
                 dataloader:CorrelationCoefficientGaussianLoader=None,
                 binary_classifier:BaseBinaryClassifier=None):

        self.config = config
        self.learning_rate = config.trainer.learning_rate
        self.number_of_epochs = config.trainer.number_of_epochs
        self.device = torch.device(config.trainer.device)
        self.loss_type = config.trainer.loss_type

        if self.loss_type == "mine":
            self.loss = mine_loss
        elif self.loss_type == "contrastive":
            self.loss = contrastive_loss

        if binary_classifier is not None:
            self.dataloader = dataloader
            self.binary_classifier = binary_classifier
            self.binary_classifier.to(self.device)
        else:
            DMINE = DynamicMutualInformationNaiveEstimator()
            DMINE.create_new_from_config(self.config,self.device)
            self.dataloader = DMINE.dataloader
            self.classifier_per_time = DMINE.classifier_per_time

    def parameters_info(self):
        print("# ==================================================")
        print("# START OF BACKWARD MI TRAINING ")
        print("# ==================================================")
        print("# Binary Classifier *********************************")
        pprint(asdict(self.binary_classifier.config))
        print("# Paths Parameters **********************************")
        pprint(asdict(self.dataloader.config))
        print("# Trainer Parameters")
        pprint(asdict(self.config))
        print("# ==================================================")
        print("# Number of Epochs {0}".format(self.number_of_epochs))
        print("# ==================================================")

    def preprocess_data(self,data_batch):
        data_batch["join"] = data_batch["join"].to(self.device)
        data_batch["independent"] = data_batch["independent"].to(self.device)
        return data_batch

    def take_time_batch(self,data_batch,time_index):
        join_timeseries = data_batch["join"]
        independent_timeseries = data_batch["independent"]
        time_data_batch = {"join": join_timeseries[:, [0, time_index]],
                           "independent": independent_timeseries[:, [0, time_index]]}
        return time_data_batch

    def train_step(self,data_batch,number_of_training_step):
        losses = []
        for time_index in range(self.dataloader.number_of_time_steps):
            data_batch = self.preprocess_data(data_batch)
            time_data_batch = self.take_time_batch(data_batch,time_index)

            binary_classifier = self.classifier_per_time[time_index]
            loss = self.loss(time_data_batch, binary_classifier)

            self.optimizers[time_index].zero_grad()
            loss.backward()
            self.optimizers[time_index].step()

            self.writer.add_scalar('classifier{0}/training loss'.format(time_index), loss, number_of_training_step)
            losses.append(loss)
        return losses

    def test_step(self,data_batch):
        with torch.no_grad():
            losses = []
            for time_index in range(self.dataloader.number_of_time_steps):
                data_batch = self.preprocess_data(data_batch)
                time_data_batch = self.take_time_batch(data_batch, time_index)

                binary_classifier = self.classifier_per_time[time_index]
                loss = self.loss(time_data_batch, binary_classifier)

                losses.append(loss)
            return losses

    def initialize(self):
        from torch.utils.tensorboard import SummaryWriter
        data_batch = next(self.dataloader.train().__iter__())
        data_batch = self.preprocess_data(data_batch)

        self.writer = SummaryWriter(self.config.experiment_files.tensorboard_path)
        self.optimizers = []
        initial_losses = []
        for time_index in range(self.dataloader.number_of_time_steps):
            binary_classifier = self.classifier_per_time[time_index]
            self.optimizers.append(Adam(binary_classifier.parameters(),lr=self.learning_rate))
            time_data_batch = self.take_time_batch(data_batch,time_index)
            initial_loss = self.loss(time_data_batch, binary_classifier)
            assert torch.isnan(initial_loss).any() == False
            assert torch.isinf(initial_loss).any() == False

            initial_losses.append(initial_loss)

        self.save_results(self.classifier_per_time,
                          initial_losses,
                          None,
                          None,
                          None,
                          0,
                          checkpoint=True)

        return initial_losses

    def train(self):
        initial_loss = self.initialize()
        best_loss = initial_loss[1]

        number_of_training_step = 0
        number_of_test_step = 0
        for epoch in range(self.number_of_epochs):

            LOSS = []
            train_loss = []
            for data_batch in self.dataloader.train():
                losses = self.train_step(data_batch,number_of_training_step)
                loss = losses[1]
                train_loss.append(loss.item())
                LOSS.append(loss.item())
                number_of_training_step += 1
            average_train_loss = np.asarray(train_loss).mean()

            test_loss = []
            for data_batch in self.dataloader.test():
                losses = self.test_step(data_batch)
                loss = losses[1]
                test_loss.append(loss.item())
                number_of_test_step+=1
            average_test_loss = np.asarray(test_loss).mean()

            # SAVE RESULTS IF LOSS DECREASES IN VALIDATION
            if average_test_loss < best_loss:
                self.save_results(self.classifier_per_time,
                                  initial_loss,
                                  average_train_loss,
                                  average_test_loss,
                                  LOSS,
                                  epoch,
                                  checkpoint=False)

            if (epoch + 1) % self.config.trainer.save_model_epochs == 0:
                self.save_results(self.classifier_per_time,
                                  initial_loss,
                                  average_train_loss,
                                  average_test_loss,
                                  LOSS,
                                  epoch+1,
                                  checkpoint=True)

            if epoch % 2 == 0:
                print("Epoch: {}, Loss: {}".format(epoch + 1, average_train_loss))

        self.writer.close()

    def save_results(self,
                     classifier_per_time,
                     initial_losses,
                     average_train_loss,
                     average_test_loss,
                     LOSS,
                     epoch=0,
                     checkpoint=False):
        if checkpoint:
            RESULTS = {
                "classifier_per_time":classifier_per_time,
                "initial_loss":initial_losses,
                "average_train_loss":average_train_loss,
                "average_test_loss":average_test_loss,
                "LOSS":LOSS
            }
            torch.save(RESULTS,self.config.experiment_files.best_model_path_checkpoint.format(epoch))
        else:
            RESULTS = {
                "classifier_per_time":classifier_per_time,
                "initial_losses":initial_losses,
                "average_train_loss":average_train_loss,
                "average_test_loss":average_test_loss,
                "LOSS":LOSS
            }
            torch.save(RESULTS,self.config.experiment_files.best_model_path)