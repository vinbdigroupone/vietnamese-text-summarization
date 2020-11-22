import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="",
                        help="config yaml path")
    parser.add_argument("--load", type=str, default="",
                        help="path to model weight")
    parser.add_argument("--mode", type=str, default="train",
                        help="model running mode (train/valid/test)")

    args = parser.parse_args()

    return args
