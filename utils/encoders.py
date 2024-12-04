"""Module containing encoder classes for data preprocessing."""

from typing import Optional
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer


class SequenceEncoder:
    """Encodes raw column strings into embeddings using a sentence transformer."""

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None
    ) -> None:
        """Initialize the SequenceEncoder.

        Args:
            model_name (str): Name of the transformer model to use. Defaults to "all-MiniLM-L6-v2".
            device (str, optional): Device to run the model on. Defaults to None.
        """
        self.device: str = device or ""
        self.model = SentenceTransformer(model_name, device=self.device)

    @torch.no_grad()
    def __call__(self, df: pd.DataFrame) -> torch.Tensor:
        """Encode the DataFrame values into embeddings.

        Args:
            df (pd.DataFrame): Data containing the strings to encode.

        Returns:
            torch.Tensor: Encoded embeddings as a tensor.
        """
        x = self.model.encode(
            df.values,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device,
        )
        return x.cpu()


class TagEncoder:
    """Converts raw column strings into categorical labels by splitting and encoding."""

    def __init__(self, sep: str = "|") -> None:
        """Initialize the TagEncoder.

        Args:
            sep (str): Separator used to split the strings. Defaults to "|".
        """
        self.sep: str = sep

    def __call__(self, df: pd.DataFrame) -> torch.Tensor:
        """Encode the DataFrame values into one-hot encoded tensors.

        Args:
            df (pd.DataFrame): Data containing the strings to encode.

        Returns:
            torch.Tensor: One-hot encoded tensor representing the tags.
        """
        tags = {g for col in df.values for g in col.split(self.sep)}
        mapping = {tag: i for i, tag in enumerate(tags)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for label in col.split(self.sep):
                x[i, mapping[label]] = 1
        return x


class IdentityEncoder:
    """Converts raw column values to PyTorch tensors."""

    def __init__(self, dtype: Optional[torch.dtype] = None) -> None:
        """Initialize the IdentityEncoder.

        Args:
            dtype (torch.dtype, optional): Data type of the output tensor. Defaults to None.
        """
        self.dtype: Optional[torch.dtype] = dtype

    def __call__(self, df: pd.DataFrame) -> torch.Tensor:
        """Convert the DataFrame values to a tensor.

        Args:
            df (pd.DataFrame): Data to convert.

        Returns:
            torch.Tensor: Tensor containing the data.
        """
        return torch.from_numpy(df.values).view(-1, 1).to(dtype=self.dtype)
