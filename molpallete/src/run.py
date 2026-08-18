import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch.multiprocessing as _mp
_mp.set_sharing_strategy('file_system')

import hydra
from omegaconf import DictConfig
from main import train, test
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    if config.run_mode == 'train':
        train(config)
    elif config.run_mode == 'test':
        test(config)


if __name__ == '__main__':
    main()
