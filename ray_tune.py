from functools import partial

import os
import sys
sys.path.insert(0, '/tank/local/ndf3868/GODDS/GAN')

import GODDS_GAN

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

def tune_params(config):
    GODDS_GAN.run_experiment(config)

if __name__ == "__main__":
    config = {
        "lr_gen": tune.loguniform(1e-4, 1e-2),
        "lr_dis": tune.loguniform(1e-4, 1e-2),
        "penalty": tune.choice([i for i in range(4, 12+1)]),
        "beta1": tune.loguniform(0.5, 0.999),
        "beta2": tune.loguniform(0.995, 0.999),
        "noise_size": tune.choice([50, 70, 100])
    }

    scheduler = ASHAScheduler(
        metric="combined",
        mode="min",
        grace_period=1,
        reduction_factor=2,
    )

    result = tune.run(
        partial(tune_params),
        resources_per_trial={"cpu": 13, "gpu": 1},
        config=config,
        num_samples=10,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("combined", "min", "last")
    print(f"Best trial config: {best_trial.config}")