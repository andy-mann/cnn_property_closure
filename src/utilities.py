import argparse
import json


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("-c", "--conditional", help="Conditional network True/False")

    parser.add_argument("-pc", "--pc_scores", help="Number of PC scores to train on")

    parser.add_argument(
        "--dir", default="/Users/andrew/Documents/PhD/Code/CondNF_PropertyClosure"
    )
    parser.add_argument("-l", "--layers", help="Number of invertible layers")
    parser.add_argument("--epochs", help="number of epochs")
    parser.add_argument("--batch_size")
    parser.add_argument("--config")


def get_config(argv):
    parser = argparse.ArgumentParser()
    add_args(parser)

    args = parser.parse_args(argv)
    config_str = args.config
    with open(config_str) as f:
        config = json.load(f)

    return config
