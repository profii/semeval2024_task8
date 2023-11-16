import argparse


def get_args():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    return args

def add_args(parser: argparse.ArgumentParser):

    parser.add_argument('--model_name',
                        type=str,
                        default="bert-base-uncased")
