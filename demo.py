"""Main entry point for the POC demo"""

import argparse
from model.dataset import BNNHDataSet
from model.han import BNNHANPOC
from config.includes import BNNHDSDIR

def build_dataset(params):
    '''PyTorch Geometry GNN dataset build utils'''
    print("PyTorch Geometry BNNHDataSet dataset build utils:", params)
    if params == '{}':
        db = BNNHDataSet(root=BNNHDSDIR)
        print("BNNHDataSet:", db._data)
    elif params == 'force':
        db = BNNHDataSet(root=BNNHDSDIR, force_rebuild=True)
        print("BNNHDataSet:", db._data)
    else:
        print("PyTorch Geometry BNNHDataSet dataset build utils")

def run_poc(params):
    '''PyTorch Geometry GNN POC demos'''
    print("PyTorch Geometry POC demos with BNNHDataSet:", params)
    if params == '{}':
        poc = BNNHANPOC()
        poc.run()
    else:
        print("PyTorch Geometry POC demos with BNNHDataSet")

def parse_args():
    """CLI Argument parser for the application"""
    parser = argparse.ArgumentParser(description="BNN HAN Model POC Demo Utilities")
    subparser = parser.add_subparsers(dest="command")

    tester = subparser.add_parser("demo", help="BRAINGNNet POC/demos")
    builder = subparser.add_parser("build", help="BNNHDS Hetero DataSet build utilities")
    builder.add_argument('--force', help='Re-build BNNDDS Hetero Dataset from raw data', type=str, required=False)

    convertgroup1 = tester.add_mutually_exclusive_group(required=True)
    convertgroup1.add_argument("--base", help="Node-Classification HAN implementation Demo", nargs="?", const="{}", type=str)
    convertgroup1.add_argument("--adv", help="[Enhanced] HAN implementation Demo", nargs="?", const="{}", type=str)

    args = parser.parse_args()
    return args


def run_session(args):
    """Run session for the application"""
    # pprint.pprint(args)
    if args.command is None:
        print("Undefined utility command. Options: demo, build")
    elif args.command == "demo" and args.base is not None:
        run_poc(args.base)
    elif args.command == "demo" and args.adv is not None:
        print("Currently unimplemented:", args.adv)
    elif args.command == "build" and args.base is not None:
        build_dataset(args.force)
    else:
        print("Unknown command option for POC: ", args.command)
    print("\nEnd of demo session...", args.command)


if __name__ == "__main__":
    builder_args = parse_args()
    run_session(builder_args)
