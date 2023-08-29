from dataclasses import dataclass

@dataclass
class MITrainerConfig:
    name:str = "MutualInformationTrainer"
    learning_rate: float = 1e-3
    number_of_epochs: int = 1000
    save_model_epochs:int = 100
    loss_type:str = "contrastive" #contrastive,mine
    experiment_class: str = "multivariate_gaussian"
    device:str = "cuda:0"

@dataclass
class DynamicNaiveMITrainerConfig:
    name:str = "DynamicMutualInformationTrainerNaive"
    learning_rate: float = 1e-3
    number_of_epochs: int = 1000
    save_model_epochs:int = 100
    loss_type:str = "contrastive" #contrastive,mine
    device:str = "cuda:0"
