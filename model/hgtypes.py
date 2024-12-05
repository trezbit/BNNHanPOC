"""Module for the BrainNet HGNN Model definitions."""

from enum import Enum

# MINIMAL EEG SPAN TOPOLOGY REPRESENTATION


class NodeType(Enum):
    """Model node type enumeration."""

    def __init__(self, label: str, index_col: str) -> None:
        """Initialize the NodeType.

        Args:
            label (str): Label of the node type.
            index_col (str): Index column name.
        """
        self.label: str = label
        self.index_col: str = index_col

    # NODES
    READ_LOC = ("READ_LOC", "readlocId")
    SUBJECT = ("SUBJECT", "subjectId")
    WAVE_ABP = ("WAVE_ABP", "waveABPId")


class RelationType(Enum):
    """Model relation type enumeration."""

    def __init__(self, label: str, bidirectional: bool = False) -> None:
        """Initialize the RelationType.

        Args:
            label (str): Label of the relation type.
            bidirectional (bool, optional): Indicates if the relation is bidirectional. Defaults to False.
        """
        self.label: str = label
        self.bidirectional: bool = bidirectional

    # Directed relations
    HAS_READ = ("HAS_READ", False)
    HAS_ABP = ("HAS_ABP", False)
