
from dcgan import Generator, Discriminator
from ..data_utils import CatDataset


def main(config: dict):
    print("config:", config)

    config


if __name__ == "__main__":
    import argparse
    import os
    import json

    parser = argparse.ArgumentParser(description="Train a DCGAN model")
    parser.add_argument('--config', type=str,
                        default=os.path.abspath(os.path.join(__file__, "..", "config.json")),
                        help='Path to the configuration file')
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config, 'r') as f:
        config = json.loads(f.read())

    main(config)