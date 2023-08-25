from typing import Union
from mutual_information.configs.mi_config import MutualInformationConfig
from mutual_information.models.binary_classifiers import BaseBinaryClassifier
from mutual_information.models.binary_classifier_config import BaseBinaryClassifierConfig

def load_binary_classifier(config:Union[BaseBinaryClassifierConfig,MutualInformationConfig]):
    if isinstance(config,MutualInformationConfig):
        config_ = config.binary_classifier
    elif isinstance(config, BaseBinaryClassifierConfig):
        config_ = config
    else:
        raise Exception("No Classifier Config Found")

    if config_.name == "BaseBinaryClassifier":
        binary_classifier = BaseBinaryClassifier(config_)
    else:
        raise Exception("No Classifier")
    return binary_classifier