import os.path as osp
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.transforms import ToUndirected
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from typing import Callable, List, Optional

from model.hgtypes import NodeType, RelationType
from utils.encoders import SequenceEncoder, TagEncoder, IdentityEncoder


class BNNHDataSet(InMemoryDataset):
    r"""BRAINGNNet Data Set from:  https://osf.io/pfx32/
    BNNDB is a heterogeneous graph containing three types of entities - SUBJECT
    (166 nodes), READ_LOC (8,727 nodes), and WAVE_ABP (19,274 nodes).
    The SUBJECTS are divided into two classes according to chronic pain status.

    Args:
        root (str): Root READ_LOC where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

        RAW DATA available for download at: https://osf.io/ge27r/
    """
    url = 'https://files.osf.io/v1/resources/rsg4h/providers/osfstorage/6737a3a9f4fc990bb8b284fd'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        ignoredegree: bool = True
    ) -> None:
        self.ignoredegree = ignoredegree
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'nSUBJECT.csv', 'nREAD_LOC.csv', 'nWAVE_ABP.csv',
            'eHAS_READ.csv', 'eHAS_ABP.csv'
            'train_test_val.csv', 'labels.csv'
        ]

    @property
    def processed_file_names(self) -> str:
        return 'bnnhandata.pt'

    def download(self) -> None:
        '''Download the dataset'''
        print("Bypassed Downloading the dataset -- Using raw files @ GitHub Repo")
        return


    def process(self) -> None:
        import scipy.sparse as sp
        data = HeteroData()
        '''Processing the dataset'''
        print("Processing the dataset")
        node_index_col = 'ID'
        edge_src_index_col = 'fromId'
        edge_dst_index_col = 'toId'


        # Nodes: SUBJECT, READ_LOC, WAVE_ABP
        # SUBJECT -  ID,degree
        # 1001,27
        subject_x, subject_mapping = self._get_node(osp.join(self.raw_dir, 'nSUBJECT.csv'), index_col=node_index_col
        , encoders={
                'degree': IdentityEncoder(dtype=torch.float)
        })
        # READ_LOC - ID,channelType,lattitude,cm0,cm1,refx,refy,absdist,brainRegTags
        # 10001,102,1,0,1,0,0,1,FrontalPolar|LEFT
        read_loc_x, read_loc_mapping = self._get_node(osp.join(self.raw_dir, 'nREAD_LOC.csv'),index_col=node_index_col
        , encoders={
                'channelType': IdentityEncoder(dtype=torch.int16),
                'brainRegTags': TagEncoder(),
                'lattitude': IdentityEncoder(dtype=torch.float),
                'cm0': IdentityEncoder(dtype=torch.float),
                'cm1': IdentityEncoder(dtype=torch.float),
                'refx': IdentityEncoder(dtype=torch.bool),
                'refy': IdentityEncoder(dtype=torch.bool),
                'absdist': IdentityEncoder(dtype=torch.float)
        })
        # WAVE_ABP - ID,waveType,adWeight
        # 110001,10,83.79255721915052
        wave_abp_x, wave_abp_mapping = self._get_node(osp.join(self.raw_dir, 'nWAVE_ABP.csv'), index_col=node_index_col
        , encoders={
                'waveType': IdentityEncoder(dtype=torch.int16),
                'adWeight': IdentityEncoder(dtype=torch.float32)
        })
        # Edges: HAS_READ, HAS_ABP
        # HAS_READ - fromId,toId
        # HAS_ABP - fromId,toId
        has_read_index, has_read_label = self._get_edge(
            osp.join(self.raw_dir, 'eHAS_READ.csv')
            , src_index_col=edge_src_index_col, src_mapping=subject_mapping
            , dst_index_col=edge_dst_index_col, dst_mapping=read_loc_mapping
            , encoders=None)

        has_abp_index, has_abp_label = self._get_edge(
            osp.join(self.raw_dir, 'eHAS_ABP.csv')
            , src_index_col=edge_src_index_col, src_mapping=read_loc_mapping
            , dst_index_col=edge_dst_index_col, dst_mapping=wave_abp_mapping, encoders=None)

        # Add SUBJECT node features for message passing:
        if (self.ignoredegree):
            data[NodeType.SUBJECT.label].x = torch.zeros(len(subject_mapping), 1)
        else:
            data[NodeType.SUBJECT.label].x = subject_x

        # Add READ_LOC node features for message passing:
        data[NodeType.READ_LOC.label].x = read_loc_x
        # Add WAVE_ABP node features
        data[NodeType.WAVE_ABP.label].x = wave_abp_x

         # Add READS for SUBJECTS at LOCATIONS
        data[NodeType.SUBJECT.label, RelationType.HAS_READ.label, NodeType.READ_LOC.label].edge_index = has_read_index
        data[NodeType.SUBJECT.label, RelationType.HAS_READ.label, NodeType.READ_LOC.label].edge_label = has_read_label

        # HIGHLIGHT for READS at LOCATIONS TO WAVE ABP
        data[NodeType.READ_LOC.label, RelationType.HAS_ABP.label, NodeType.WAVE_ABP.label].edge_index = has_abp_index
        data[NodeType.READ_LOC.label, RelationType.HAS_ABP.label, NodeType.WAVE_ABP.label].edge_label = has_abp_label

        # Add a reverse ('SUBJECT', 'rev_rates', 'user') relation for message passing.
        data = ToUndirected()(data)
        del data[NodeType.READ_LOC.label, 'rev_' + RelationType.HAS_READ.label, NodeType.SUBJECT.label].edge_label  # Remove "reverse" label.
        del data[NodeType.WAVE_ABP.label, 'rev_' + RelationType.HAS_ABP.label, NodeType.READ_LOC.label].edge_label  # Remove "reverse" label.
        y = self._get_labels(osp.join(self.raw_dir, 'nlabels.csv'))
        data['SUBJECT'].y = y

        train_mask, test_mask, val_mask = self._get_train_test_val_mask(osp.join(self.raw_dir, 'ntrain_test_val.csv'))
        data['SUBJECT'].train_mask = train_mask
        data['SUBJECT'].test_mask = test_mask
        data['SUBJECT'].val_mask = val_mask
        #print("After Masks: ", data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def _get_node(self, path, index_col, encoders=None, **kwargs):
        df = pd.read_csv(path, index_col=index_col, **kwargs)
        mapping = {index: i for i, index in enumerate(df.index.unique())}
        x = None
        if encoders is not None:
            xs = [encoder(df[col]) for col, encoder in encoders.items()]
            x = torch.cat(xs, dim=-1)
        return x, mapping
    def _get_edge(self, path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                    encoders=None, **kwargs):
        df = pd.read_csv(path)

        src = [src_mapping[index] for index in df[src_index_col]]
        dst = [dst_mapping[index] for index in df[dst_index_col]]
        edge_index = torch.tensor([src, dst])
        edge_attr = None
        if encoders is not None:
            edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
            edge_attr = torch.cat(edge_attrs, dim=-1)
        return edge_index, edge_attr

    def _get_labels(self, path):
        df = pd.read_csv(path, index_col='subjectId')
        col = 'label'
        y = torch.tensor(df[col].values, dtype=torch.long)
        return y

    def _get_train_test_val_mask(self, path):
        df = pd.read_csv(path, index_col='subjectId')
        train_mask = None
        test_mask = None
        val_mask = None

        train_mask = torch.tensor(df['train'].values, dtype=torch.bool)
        test_mask = torch.tensor(df['test'].values, dtype=torch.bool)
        val_mask = torch.tensor(df['val'].values, dtype=torch.bool)

        return train_mask, test_mask, val_mask

