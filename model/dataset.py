"""
This module defines the `BNNHDataSet` class, which provides a PyTorch Geometric
heterogeneous dataset for EEG-based chronic pain detection using graph neural networks.

The dataset integrates EEG data from chronic pain patients and healthy controls,
transforming them into a graph structure that aligns brain topology with EEG electrode
placements and Absolute Band Power (ABP) features across different frequency bands.

Node Types:
- SUBJECT: Represents individual participants with labels indicating chronic pain status.
- READ_LOC: Represents EEG read locations corresponding to electrode positions.
- WAVE_ABP: Represents the ABP features extracted from EEG signals.

Edge Types:
- HAS_READ: Connects SUBJECT nodes to READ_LOC nodes.
- HAS_ABP: Connects READ_LOC nodes to WAVE_ABP nodes.

The `BNNHDataSet` class processes the raw data files and constructs a `HeteroData` object
with all node features, edge indices, and labels necessary for training a Heterogeneous
Graph Attention Network (HAN) model.

Datasets Used:
- cpCGX-BIDS: EEG data from chronic pain patients.
- MBB LEMON: EEG data from healthy control subjects.

The dataset is designed to facilitate research in EEG-based chronic pain detection by
providing a ready-to-use graph structure compatible with PyTorch Geometric.
"""

import os.path as osp
from typing import Callable, Optional, Union, List, Tuple, Dict

import pandas as pd
import torch
from torch_geometric.transforms import ToUndirected
from torch_geometric.data import HeteroData, InMemoryDataset


from model.hgtypes import NodeType, RelationType
from utils.encoders import TagEncoder, IdentityEncoder


class BNNHDataSet(InMemoryDataset):
    r"""
    `BNNHDataSet` is a PyTorch Geometric `InMemoryDataset` for EEG-based chronic pain detection using graph neural networks.

    The dataset integrates EEG data from chronic pain patients and healthy controls,
    transforming them into a graph structure that aligns brain topology with EEG electrode
    placements and Absolute Band Power (ABP) features across different frequency bands.

    **Node Types:**
    - **SUBJECT**: Represents individual participants with labels indicating chronic pain status.
    - **READ_LOC**: Represents EEG read locations corresponding to electrode positions.
    - **WAVE_ABP**: Represents the ABP features extracted from EEG signals.

    **Edge Types:**
    - **HAS_READ**: Connects SUBJECT nodes to READ_LOC nodes.
    - **HAS_ABP**: Connects READ_LOC nodes to WAVE_ABP nodes.

    The dataset processes raw CSV files to construct a `HeteroData` object compatible with
    PyTorch Geometric's heterogeneous graph neural network models.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (Optional[Callable], optional): A function/transform that takes in a
            `torch_geometric.data.HeteroData` object and returns a transformed version.
            The data object will be transformed before every access. Defaults to `None`.
        pre_transform (Optional[Callable], optional): A function/transform that takes in a
            `torch_geometric.data.HeteroData` object and returns a transformed version.
            The data object will be transformed before being saved to disk. Defaults to `None`.
        pre_filter (Optional[Callable], optional): A function that takes in a
            `torch_geometric.data.HeteroData` object and returns a boolean value, indicating
            whether the data object should be included in the final dataset. Defaults to `None`.
        log (bool, optional): Whether to log the progress. Defaults to `True`.
        force_reload (bool, optional): Whether to re-process the dataset even if processed files exist. Defaults to `False`.
        ignoredegree (bool, optional): Whether to ignore the 'degree' feature for SUBJECT nodes. Defaults to `True`.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
        ignoredegree: bool = True,
    ) -> None:
        """
        Initialize the `BNNHDataSet`.

        Args:
            root (str): Root directory where the dataset should be saved.
            transform (Optional[Callable], optional): Function to transform the data object before every access. Defaults to `None`.
            pre_transform (Optional[Callable], optional): Function to transform the data object before saving to disk. Defaults to `None`.
            pre_filter (Optional[Callable], optional): Function to filter data objects before saving to disk. Defaults to `None`.
            log (bool, optional): Whether to log the progress. Defaults to `True`.
            force_reload (bool, optional): Whether to re-process the dataset even if processed files exist. Defaults to `False`.
            ignoredegree (bool, optional): Whether to ignore the 'degree' feature for SUBJECT nodes. Defaults to `True`.
        """
        self.ignoredegree: bool = ignoredegree
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            log=log,
            force_reload=force_reload,
        )
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        """List of raw file names expected to be found in `self.raw_dir`.

        Returns:
            List[str]: List of file names required for processing the dataset.
        """
        return [
            "nSUBJECT.csv",
            "nREAD_LOC.csv",
            "nWAVE_ABP.csv",
            "eHAS_READ.csv",
            "eHAS_ABP.csv",
            "train_test_val.csv",
            "labels.csv",
        ]

    @property
    def processed_file_names(self) -> str:
        """Name of the processed file saved in `self.processed_dir`.

        Returns:
            str: Name of the processed data file.
        """
        return "bnnhandata.pt"

    def download(self) -> None:
        """Bypass the download process since raw data is locally available.

        Note:
            This method is overridden to avoid downloading because the raw data is expected to be present locally.
        """
        print(
            "Bypassed downloading the dataset -- Using raw files from the local repository."
        )

    def process(self) -> None:
        """Processes the raw data files and constructs the `HeteroData` object.

        This method reads node and edge CSV files, encodes features using the specified encoders,
        constructs mappings, and builds the heterogeneous graph data object compatible with PyTorch Geometric.
        """
        data = HeteroData()
        print("Processing the dataset")
        node_index_col: str = "ID"
        edge_src_index_col: str = "fromId"
        edge_dst_index_col: str = "toId"

        # Nodes: SUBJECT, READ_LOC, WAVE_ABP
        subject_x, subject_mapping = self._get_node(
            osp.join(self.raw_dir, "nSUBJECT.csv"),
            index_col=node_index_col,
            encoders={"degree": IdentityEncoder(dtype=torch.float)},
        )
        read_loc_x, read_loc_mapping = self._get_node(
            osp.join(self.raw_dir, "nREAD_LOC.csv"),
            index_col=node_index_col,
            encoders={
                "channelType": IdentityEncoder(dtype=torch.int16),
                "brainRegTags": TagEncoder(),
                "lattitude": IdentityEncoder(dtype=torch.float),
                "cm0": IdentityEncoder(dtype=torch.float),
                "cm1": IdentityEncoder(dtype=torch.float),
                "refx": IdentityEncoder(dtype=torch.bool),
                "refy": IdentityEncoder(dtype=torch.bool),
                "absdist": IdentityEncoder(dtype=torch.float),
            },
        )
        wave_abp_x, wave_abp_mapping = self._get_node(
            osp.join(self.raw_dir, "nWAVE_ABP.csv"),
            index_col=node_index_col,
            encoders={
                "waveType": IdentityEncoder(dtype=torch.int16),
                "adWeight": IdentityEncoder(dtype=torch.float32),
            },
        )
        # Edges: HAS_READ, HAS_ABP
        has_read_index, has_read_label = self._get_edge(
            osp.join(self.raw_dir, "eHAS_READ.csv"),
            src_index_col=edge_src_index_col,
            src_mapping=subject_mapping,
            dst_index_col=edge_dst_index_col,
            dst_mapping=read_loc_mapping,
            encoders=None,
        )

        has_abp_index, has_abp_label = self._get_edge(
            osp.join(self.raw_dir, "eHAS_ABP.csv"),
            src_index_col=edge_src_index_col,
            src_mapping=read_loc_mapping,
            dst_index_col=edge_dst_index_col,
            dst_mapping=wave_abp_mapping,
            encoders=None,
        )

        # Add SUBJECT node features for message passing:
        if self.ignoredegree:
            data[NodeType.SUBJECT.label].x = torch.zeros(len(subject_mapping), 1)
        else:
            data[NodeType.SUBJECT.label].x = subject_x

        # Add READ_LOC node features for message passing:
        data[NodeType.READ_LOC.label].x = read_loc_x
        # Add WAVE_ABP node features
        data[NodeType.WAVE_ABP.label].x = wave_abp_x

        # Add READS for SUBJECTS at LOCATIONS
        data[
            NodeType.SUBJECT.label, RelationType.HAS_READ.label, NodeType.READ_LOC.label
        ].edge_index = has_read_index
        data[
            NodeType.SUBJECT.label, RelationType.HAS_READ.label, NodeType.READ_LOC.label
        ].edge_label = has_read_label

        # HIGHLIGHT for READS at LOCATIONS TO WAVE ABP
        data[
            NodeType.READ_LOC.label, RelationType.HAS_ABP.label, NodeType.WAVE_ABP.label
        ].edge_index = has_abp_index
        data[
            NodeType.READ_LOC.label, RelationType.HAS_ABP.label, NodeType.WAVE_ABP.label
        ].edge_label = has_abp_label

        # Add a reverse relation for message passing.
        data = ToUndirected()(data)
        del data[
            NodeType.READ_LOC.label,
            "rev_" + RelationType.HAS_READ.label,
            NodeType.SUBJECT.label,
        ].edge_label  # Remove "reverse" label.
        del data[
            NodeType.WAVE_ABP.label,
            "rev_" + RelationType.HAS_ABP.label,
            NodeType.READ_LOC.label,
        ].edge_label  # Remove "reverse" label.
        y = self._get_labels(osp.join(self.raw_dir, "nlabels.csv"))
        data["SUBJECT"].y = y

        train_mask, test_mask, val_mask = self._get_train_test_val_mask(
            osp.join(self.raw_dir, "ntrain_test_val.csv")
        )
        data["SUBJECT"].train_mask = train_mask
        data["SUBJECT"].test_mask = test_mask
        data["SUBJECT"].val_mask = val_mask

        if self.pre_transform is not None:
            data = self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        """String representation of the dataset.

        Returns:
            str: Name of the dataset class.
        """
        return f"{self.__class__.__name__}()"

    def _get_node(
        self,
        path: str,
        index_col: str,
        encoders: Optional[Dict[str, Callable[[pd.Series], torch.Tensor]]] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[int, int]]:
        """Processes node data from a CSV file.

        Args:
            path (str): Path to the CSV file.
            index_col (str): Column to use as the index.
            encoders (Optional[Dict[str, Callable[[pd.Series], torch.Tensor]]], optional): A dictionary mapping column names to encoder functions that convert pandas Series to torch Tensors. Defaults to `None`.

        Returns:
            Tuple[Optional[torch.Tensor], Dict[int, int]]: Node features and mapping.
        """

        df = pd.read_csv(path, index_col=index_col)
        mapping: Dict[int, int] = {
            index: i for i, index in enumerate(df.index.unique())
        }
        x: Optional[torch.Tensor] = None
        if encoders is not None:
            # Encode specified columns and concatenate features
            xs = [encoder(df[col]) for col, encoder in encoders.items()]
            x = torch.cat(xs, dim=-1)
        return x, mapping

    def _get_edge(
        self,
        path: str,
        src_index_col: str,
        src_mapping: Dict[int, int],
        dst_index_col: str,
        dst_mapping: Dict[int, int],
        encoders: Optional[Dict[str, Callable[[pd.Series], torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Processes edge data from a CSV file.

        Args:
            path (str): Path to the CSV file.
            src_index_col (str): Column name for source node indices.
            src_mapping (Dict[int, int]): Mapping from original source node indices to new indices.
            dst_index_col (str): Column name for destination node indices.
            dst_mapping (Dict[int, int]): Mapping from original destination node indices to new indices.
            encoders (Optional[Dict[str, Callable[[pd.Series], torch.Tensor]]], optional): Encoders for specific columns. Defaults to `None`.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Edge indices and edge attributes.
        """

        df = pd.read_csv(path)

        # Map original indices to new sequential indices for source and destination nodes
        src = [src_mapping[index] for index in df[src_index_col]]
        dst = [dst_mapping[index] for index in df[dst_index_col]]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        edge_attr: Optional[torch.Tensor] = None
        if encoders is not None:
            # Encode specified columns and concatenate edge attributes
            edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
            edge_attr = torch.cat(edge_attrs, dim=-1)
        return edge_index, edge_attr

    def _get_labels(self, path: str) -> torch.Tensor:
        """Loads node labels from a CSV file.

        Args:
            path (str): Path to the CSV file containing labels.

        Returns:
            torch.Tensor: Tensor containing the labels for SUBJECT nodes.
        """
        df = pd.read_csv(path, index_col="subjectId")
        y = torch.tensor(df["label"].values, dtype=torch.long)
        return y

    def _get_train_test_val_mask(
        self, path: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reads train, test, and validation masks from a CSV file.

        Args:
            path (str): Path to the CSV file containing the masks.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tensors containing the train, test, and validation masks.
        """
        df = pd.read_csv(path, index_col="subjectId")
        train_mask = torch.tensor(df["train"].values, dtype=torch.bool)
        test_mask = torch.tensor(df["test"].values, dtype=torch.bool)
        val_mask = torch.tensor(df["val"].values, dtype=torch.bool)

        return train_mask, test_mask, val_mask
