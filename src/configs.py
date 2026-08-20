from dataclasses import dataclass

@dataclass
class ALState:
    labeled: list[int]
    unlabeled: list[int]

@dataclass
class DataConfig:
    train_size: float = 0.9
    val_size: float = 0.1
    test_size: float = 0.0
    batch_size: int = 1024
    split: str = "perc"  

@dataclass
class TrainConfig:
    use_al: bool = True
    starting_epoch_multiplier: int = 9 
    lr: float =  0.00063
    weight_decay: float = 2.3e-05 
    epochs: int = 11
    loss_weight: float = 0.811
    num_runs: int = 1
   
@dataclass
class ModelConfig:
    depth_conv: int = 5 
    depth_ffnn: int = 3
    hidden_channels: int = 512 
    dropout_rate: float = 0.37188
    patience: int = 20
    
@dataclass
class ActiveLearningConfig:
    acquisition_budget_total: float = 96 * 10
    rounds: int = 10
    alpha: float = 0.38  
    new_query_weight: float = 7.0


@dataclass
class Config:
    data_cfg: DataConfig
    train_cfg: TrainConfig
    model_cfg: ModelConfig
    al_cfg: ActiveLearningConfig

NON_AL_TRAIN_DEFAULTS: dict[str, float] = {
    "lr":   0.00038136563138314996,
    'weight_decay': 8.165696312849013e-06,
    "epochs": 72,
    'starting_epoch_multiplier': 1,
}

NON_AL_MODEL_DEFAULTS: dict[str, float] = {
    'dropout_rate': 0.13249619898782194,
    'hidden_channels': 512,
    'depth_conv': 4, 
    'depth_ffnn': 4
}