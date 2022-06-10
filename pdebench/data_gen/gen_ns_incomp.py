#!/usr/bin/env python

import hydra
import dotenv
from omegaconf import DictConfig, OmegaConf

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv()


@hydra.main(config_path="configs/", config_name="ns_incomp.yaml")
def main(config: DictConfig):
    """
    This is a starter function of the simulation

    Args:
        config: This function uses hydra configuration for all parameters.
    """
    
    from src import sim_ns_incomp_2d
    sim_ns_incomp_2d.ns_sim(config=config, **config)


if __name__ == "__main__":
    main()
