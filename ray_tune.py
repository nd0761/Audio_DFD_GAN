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
        "lr_gen": tune.loguniform(1e-4/4, 1e-4*3),
        "lr_dis": tune.loguniform(1e-4/4, 1e-4*3),
        "penalty": tune.choice([0, 3, 6, 9]),
        # "beta1": tune.loguniform(0.5, 0.999),
        # "beta2": tune.loguniform(0.9, 0.999),
        "g_trainin_step": tune.choice([10, 30, 60, 90]),
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
        num_samples=25,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("combined", "min", "last")
    print(f"Best trial config: {best_trial.config}")