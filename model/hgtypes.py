'''Module for the BrainNet HGNN Model definitions.'''
from enum import Enum

# MINIMAL EEG SPAN TOPOLOGY REPRESENTATION

# Enumerations for the model
class NodeType(Enum):
    '''Model Node type enumeration'''
    def __init__(self, label, index_col):
        self.label = label
        self.index_col = index_col

    ## NODES
    READ_LOC='READ_LOC', 'readlocId'
    SUBJECT='SUBJECT', 'subjectId'
    WAVE_ABP='WAVE_ABP', 'waveABPId'


class RelationType(Enum):
    '''Model relation type enumeration'''
    def __init__(self, label, birectional=False, src='fromId', dst='toId'):
        self.label = label
        self.birectional = birectional
        self.srcId = src
        self.dstId = dst
    # Directed relations
    HAS_READ='HAS_READ',  False
    HAS_ABP='HAS_ABP', False


