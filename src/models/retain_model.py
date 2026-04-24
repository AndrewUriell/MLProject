from typing import Dict, Optional
 
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
 
from pyhealth.models.base_model import BaseModel
from pyhealth.models.embedding import EmbeddingModel
 
 
# Attention layer for an individual feature stream (like for example conditions, procedures)
class RETAINLayer(nn.Module):
 
    def __init__(self, feature_size: int, dropout: float = 0.5, use_alpha: bool = True, use_beta: bool = True,):
        super(RETAINLayer, self).__init__()
        self.feature_size = feature_size
        self.use_alpha = use_alpha
        self.use_beta = use_beta
        self.dropout_layer = nn.Dropout(p=dropout)

        if use_alpha:
            self.alpha_gru = nn.GRU(feature_size, feature_size, batch_first=True)
            self.alpha_li = nn.Linear(feature_size, 1)

        if use_beta:
            self.beta_gru = nn.GRU(feature_size, feature_size, batch_first=True)
            self.beta_li = nn.Linear(feature_size, feature_size)
 
     # Each sequence will get flipped so process most recent visit first
    #NEED THIS AS IT DOESN'T USE SELF, CODE CRASHED WITHOUT IT
    @staticmethod
    def reverse_x(input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        reversed_input = input.new(input.size())
        for i, length in enumerate(lengths):
            reversed_input[i, :length] = input[i, :length].flip(dims=[0])
        return reversed_input
 
    # Visit-level attention (aka: which past visits matter most)
    def compute_alpha(self, rx, lengths, total_length):
        rx = rnn_utils.pack_padded_sequence(rx, lengths, batch_first=True, enforce_sorted=False)
        g, _ = self.alpha_gru(rx)
        g, _ = rnn_utils.pad_packed_sequence(g, batch_first=True, total_length=total_length)
        return torch.softmax(self.alpha_li(g), dim=1)
 
    # Code-level attention (aka: which codes within a visit matter most)
    def compute_beta(self, rx, lengths, total_length):
        rx = rnn_utils.pack_padded_sequence(rx, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.beta_gru(rx)
        h, _ = rnn_utils.pad_packed_sequence(h, batch_first=True, total_length=total_length)
        return torch.tanh(self.beta_li(h))
 
    # Will combine the alpha and beta into a single context vector for the feature
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.dropout_layer(x)
        batch_size = x.size(0)
        total_length = x.size(1)
 
        if mask is None:
            lengths = torch.full((batch_size,), total_length, dtype=torch.int64)
        else:
            lengths = torch.sum(mask.int(), dim=-1).cpu()
        lengths = lengths.clamp(min=1)
 
        rx = self.reverse_x(x, lengths)

        #For abalation study I'm making alpha and beta "toggable", if it's off, we do uniform alpha
        if self.use_alpha:
            attn_alpha = self.compute_alpha(rx, lengths, total_length)
        else:
            attn_alpha = torch.ones(batch_size, total_length, 1, device=x.device) / total_length
        
        #Same for beta, if it's off we use identity
        if self.use_beta:
            attn_beta = self.compute_beta(rx, lengths, total_length)
        else:
            attn_beta = torch.ones(batch_size, total_length, self.feature_size, device=x.device)
 
        c = attn_alpha * attn_beta * x
        return torch.sum(c, dim=1)
 
 
# Full RETAIN model
class RETAINModel(BaseModel):
 
    def __init__(self, dataset, embedding_dim: int = 128, dropout: float = 0.5, use_alpha: bool = True, use_beta: bool = True):
        super(RETAINModel, self).__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
 
        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]
 
        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
 
        self.retain = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.retain[feature_key] = RETAINLayer(feature_size=embedding_dim, dropout=dropout, use_alpha=use_alpha, use_beta=use_beta)

 
        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * embedding_dim, output_size)
 
    # Runs each feature through the RETAIN model and does classification of the output
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        patient_emb = []
        embedded = self.embedding_model(kwargs)
 
        for feature_key in self.feature_keys:
            x = embedded[feature_key]
 
            if len(x.shape) == 4:
                x = torch.sum(x, dim=2)
            elif len(x.shape) == 2:
                x = x.unsqueeze(1)
 
            mask = (x.abs().sum(dim=-1) > 0).float()
            patient_emb.append(self.retain[feature_key](x, mask))
 
        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
 
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
 
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results