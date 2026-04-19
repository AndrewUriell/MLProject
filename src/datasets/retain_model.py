from typing import Dict, Optional
 
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
 
from pyhealth.models.base_model import BaseModel
from pyhealth.models.utils import EmbeddingModel
 
 
# Attention layer for an individual feature stream (like for example conditions, procedures)
class RETAINLayer(nn.Module):
 
    def __init__(self, feature_size: int, dropout: float = 0.5):
        super(RETAINLayer, self).__init__()
        self.feature_size = feature_size
        self.dropout_layer = nn.Dropout(p=dropout)
 
        self.alpha_gru = nn.GRU(feature_size, feature_size, batch_first=True)
        self.alpha_li = nn.Linear(feature_size, 1)
 
        self.beta_gru = nn.GRU(feature_size, feature_size, batch_first=True)
        self.beta_li = nn.Linear(feature_size, feature_size)
 
    # Each sequence will get flipped so process most recent visit first
    def reverse_x(input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
 
    # Visit-level attention (aka: which past visits matter most)
    def compute_alpha(self, rx, lengths, total_length):
        raise NotImplementedError
 
    # Code-level attention (aka: which codes within a visit matter most)
    def compute_beta(self, rx, lengths, total_length):
        raise NotImplementedError
 
    # Will combine the alpha and beta into a single context vector for the feature
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError
 
 
# Full model — each feature should have one retainlayer (I believe)
class RETAINModel(BaseModel):
 
    def __init__(self, dataset, embedding_dim: int = 128, dropout: float = 0.5):
        super(RETAINModel, self).__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
 
        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]
 
        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
 
        self.retain = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.retain[feature_key] = RETAINLayer(feature_size=embedding_dim, dropout=dropout)
 
        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * embedding_dim, output_size)
 
    # Runs each feature through the RETAIN model and does classification of the output
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError