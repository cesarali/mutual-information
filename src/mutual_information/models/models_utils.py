from typing import Union
from mutual_information.configs.mi_config import MutualInformationConfig
from mutual_information.configs.dynamic_mi_naive_config import DynamicMutualInformationNaiveConfig

from mutual_information.models.binary_classifiers import BaseBinaryClassifier
from mutual_information.models.binary_classifier_config import BaseBinaryClassifierConfig

def load_binary_classifier(config:Union[BaseBinaryClassifierConfig,MutualInformationConfig,DynamicMutualInformationNaiveConfig]):
    if isinstance(config,MutualInformationConfig):
        config_ = config.binary_classifier
    elif isinstance(config, BaseBinaryClassifierConfig):
        config_ = config
    elif isinstance(config,DynamicMutualInformationNaiveConfig):
        config_ = config.binary_classifier
    else:
        raise Exception("No Classifier Config Found")
    if config_.name == "BaseBinaryClassifier":
        binary_classifier = BaseBinaryClassifier(config_)
    else:
        raise Exception("No Classifier")
    return binary_classifier