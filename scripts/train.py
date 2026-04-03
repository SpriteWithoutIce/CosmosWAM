import hydra
from omegaconf import DictConfig
from cosmos_wam.runtime import run_training


@hydra.main(config_path="../configs", config_name="train_cosmos_2b", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
