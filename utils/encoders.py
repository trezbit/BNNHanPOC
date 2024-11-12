import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer



class SequenceEncoder:
    # The 'SequenceEncoder' encodes raw column strings into embeddings.
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device or ""
        self.model = SentenceTransformer(model_name, device=self.device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        #return x.cpu()
        return x.cpu()

class TagEncoder:
    # The 'TagEncoder' splits the raw column strings by 'sep' and converts
    # individual elements to categorical labels.
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        tags = {g for col in df.values for g in col.split(self.sep)}
        mapping = {tag: i for i, tag in enumerate(tags)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for label in col.split(self.sep):
                x[i, mapping[label]] = 1
        return x

class IdentityEncoder:
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

