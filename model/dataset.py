import os.path as osp
from typing import Callable, Optional, Union, List, Tuple, Dict

import pandas as pd
import torch
from torch_geometric.transforms import ToUndirected
from torch_geometric.data import HeteroData, InMemoryDataset

from model.hgtypes import NodeType, RelationType
from utils.encoders import TagEncoder, IdentityEncoder


class BNNHDataSet(InMemoryDataset):
    r"""BRAINGNNet Data Set from:  https://osf.io/pfx32/
    BNNDB is a heterogeneous graph containing three types of entities - SUBJECT
    (166 nodes), READ_LOC (8,727 nodes), and WAVE_ABP (19,274 nodes).
    The SUBJECTS are divided into two classes according to chronic pain status.

    RAW DATA available for download at: https://osf.io/ge27r/
    """

    url: str = (
        "https://files.osf.io/v1/resources/rsg4h/providers/osfstorage/6737a3a9f4fc990bb8b284fd"
    )

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
        """Initializes the BNNHDataSet.

        Args:
            root (str): Root directory where the dataset should be saved.
            transform (Optional[Callable], optional): A function/transform that takes in a
                `torch_geometric.data.HeteroData` object and returns a transformed version.
                The data object will be transformed before every access. Defaults to None.
            pre_transform (Optional[Callable], optional): A function/transform that takes in a
                `torch_geometric.data.HeteroData` object and returns a transformed version.
                The data object will be transformed before being saved to disk. Defaults to None.
            pre_filter (Callable, optional): A function/transform that takes in a
                `torch_geometric.data.HeteroData` object and returns a boolean value, indicating
                whether the data object should be included in the final dataset. Defaults to None.
            log (bool, optional): Whether to log the progress. Defaults to True.
            force_reload (bool, optional): Whether to re-process the dataset. Defaults to False.
            ignoredegree (bool, optional): Whether to ignore the degree feature for SUBJECT nodes. Defaults to True.
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
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        """The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading.
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
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        """The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing.
        """
        return "bnnhandata.pt"

    def download(self) -> None:
        """Downloads the dataset to the :obj:`self.raw_dir` folder."""
        print("Bypassed Downloading the dataset -- Using raw files @ GitHub Repo")

    def process(self) -> None:
        """Processes the dataset to the :obj:`self.processed_dir` folder."""

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
        return f"{self.__class__.__name__}()"

    def _get_node(
        self,
        path: str,
        index_col: str,
        encoders: Optional[Dict[str, Callable]] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[int, int]]:
        """Processes node data from a CSV file.

        Args:
            path (str): Path to the CSV file.
            index_col (str): Column to use as the index.
            encoders (Optional[Dict[str, Callable]], optional): Encoders for specific columns. Defaults to None.

        Returns:
            Tuple[Optional[torch.Tensor], Dict[int, int]]: Node features and mapping.
        """

        df = pd.read_csv(path, index_col=index_col)
        mapping = {index: i for i, index in enumerate(df.index.unique())}
        x = None
        if encoders is not None:
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
        encoders: Optional[Dict[str, Callable]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Processes edge data from a CSV file.

        Args:
            path (str): Path to the CSV file.
            src_index_col (str): Column name for source node indices.
            src_mapping (Dict[int, int]): Mapping from original source node indices to new indices.
            dst_index_col (str): Column name for destination node indices.
            dst_mapping (Dict[int, int]): Mapping from original destination node indices to new indices.
            encoders (Optional[Dict[str, Callable]], optional): Encoders for specific columns. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Edge indices and edge attributes.
        """

        df = pd.read_csv(path)

        src = [src_mapping[index] for index in df[src_index_col]]
        dst = [dst_mapping[index] for index in df[dst_index_col]]
        edge_index = torch.tensor([src, dst])
        edge_attr = None
        if encoders is not None:
            edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
            edge_attr = torch.cat(edge_attrs, dim=-1)
        return edge_index, edge_attr

    def _get_labels(self, path: str) -> torch.Tensor:
        """Reads labels from a CSV file and returns them as a tensor.

        Args:
            path (str): Path to the CSV file containing labels.

        Returns:
            torch.Tensor: Tensor containing the labels.
        """
        df = pd.read_csv(path, index_col="subjectId")
        col = "label"
        y = torch.tensor(df[col].values, dtype=torch.long)
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
