"""Main entry point for the POC demo"""

import argparse
from model.dataset import BNNHDataSet
from model.han import BNNHANPOC
from config.includes import BNNHDSDIR


def build_dataset() -> None:
    """PyTorch Geometry GNN dataset build utils"""
    print("PyTorch Geometry BNNHDataSet dataset build utils")
    db = BNNHDataSet(root=BNNHDSDIR)
    # print("BNNHDataSet:", db._data)


def run_poc() -> None:
    """PyTorch Geometry GNN POC demos"""
    print("PyTorch Geometry POC demo with BNNHDataSet")
    poc = BNNHANPOC()
    poc.run()


def parse_args() -> argparse.Namespace:
    """CLI Argument parser for the application"""
    parser = argparse.ArgumentParser(description="BNN HAN Model POC Demo Utilities")
    subparser = parser.add_subparsers(dest="command", required=True)

    tester = subparser.add_parser("demo", help="BRAINGNNet POC/demos")
    builder = subparser.add_parser(
        "build", help="BNNHDS Hetero DataSet build utilities"
    )
    builder.add_argument(
        "--show",
        help="BNNDDS Hetero Dataset from raw data",
        nargs="?",
        default=False,
        const=True,
        type=bool,
    )

    tester_group = tester.add_mutually_exclusive_group(required=True)
    tester_group.add_argument(
        "--base",
        help="Node-Classification HAN implementation Demo",
        nargs="?",
        default=False,
        const=True,
        type=bool,
    )
    tester_group.add_argument(
        "--adv",
        help="[Enhanced] HAN implementation Demo",
        nargs="?",
        default=False,
        const=True,
    )

    args = parser.parse_args()
    return args


def run_session(args: argparse.Namespace) -> None:
    """Run session for the application

    Args:
        args (argparse.Namespace): CLI arguments
    """
    if args.command == "demo":
        if args.base is not None:
            run_poc()
        if args.adv is not None:
            print("Currently unimplemented:")
    elif args.command == "build":
        build_dataset()
    else:
        raise ValueError("Unknown command option for POC: ", args.command)

    print("\nEnd of demo session...", args.command)


if __name__ == "__main__":
    builder_args = parse_args()
    run_session(builder_args)
