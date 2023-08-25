from dataclasses import dataclass

@dataclass
class BaseBinaryClassifierConfig:
    name:str = "BaseBinaryClassifier"
    input_size:int = 3
    hidden_size:int = 50
    output_transformation:str = "sigmoid"
