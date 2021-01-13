import torch

# Train from scratch
TFS_CONF = {
    "opt_fn": torch.optim.Adam,
    "T": 100,
    "train_batch_size": 16,
    "test_batch_size": 4,
    "lr": 0.001,
}

# Fine Tuning
FT_CONF = {
    "opt_fn": torch.optim.Adam,
    "T": 100,
    "train_batch_size": 16,
    "test_batch_size": 4,
    "lr": 0.001,
}

# Centroid Fine Tuning
CFT_CONF = {
    "opt_fn": torch.optim.Adam,
    "T": 100,
    "train_batch_size": 16,
    "test_batch_size": 4,
    "lr": 0.001
}

# LSTM meta-learner
LSTM_CONF = {
    "opt_fn": torch.optim.Adam,
    "T": 8,
    "lr": 0.001,
    "input_size": 4,
    "hidden_size": 20,
    "grad_clip": 0.25
}

# Model-agnostic meta-learning
MAML_CONF = {
    "opt_fn": torch.optim.Adam,
    "T": 1, 
    "lr": 0.001,
    "base_lr": 0.01,
    "meta_batch_size":1
}


# Mimicking One-Step Optimizer config
MOSO_CONF = {
    "opt_fn": torch.optim.Adam,
    "T": 100, 
    "lr": 0.001,
    "act": torch.nn.ReLU(),
    "hcopt_fn": torch.optim.Adam,
    "hcopt_lr": 0.001,
    "meta_batch_size": 32
}

# TURTLE
TURTLE_CONF = {
    "opt_fn": torch.optim.Adam,
    "T": 1, # not applicable
    "lr": 0.001,
    "act": torch.nn.ReLU(),
    "beta": 0.9,
    "meta_batch_size": 1,
    "time_input": False,
    "param_lr": False,
    "decouple": None
}