import argparse
import train
import constant as const


def parse_args():
    parser = argparse.ArgumentParser(description="Normalizing flow on PHR")
    parser.add_argument("--data", type=str, required=True, help="path to data folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.DISEASE_CATEGORIES,
        required=True,
        help="category name in mvtec",
    )
    parser.add_argument(
        '--seed', default=0, type=int,
        help='random state (default: 0)'
    )
    parser.add_argument("--use-altub", action='store_true', default=False,
                        help="whether to use altub")
    parser.add_argument("--eval-all", action='store_true', default=False,
                        help="whether to use altub")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    train.train(args)