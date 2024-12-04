"""Module to define constants and config functions for the project"""

import os
from pathlib import Path

# App root
ROOT_DIR = os.path.abspath(Path(__file__).parent.parent)
DATADIR = os.path.join(ROOT_DIR, "data")
RAWCSVDIR = os.path.join(DATADIR, "raw")
BNNHDSDIR = os.path.join(DATADIR, "bnnhds")
TORCHPTDIR = os.path.join(DATADIR, "torch")

# Graph data public sets -- for cloud graph db access
PUBLIC_GRAPH_HGNN_ROOT = (
    "https://raw.githubusercontent.com/trezbit/bnn-model-builder/master/graphdb/csv"
)
