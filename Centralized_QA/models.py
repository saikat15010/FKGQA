import torch
import torch.nn as nn
from transformers import RobertaModel

class DistMult(nn.Module):
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.ent = nn.Embedding(num_entities, dim)
        self.rel = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    def score(self, h, r):
        """
        Score all entities given head and relation embeddings
        h: (batch_size, dim)
        r: (batch_size, dim)
        Returns: (batch_size, num_entities)
        """
        x = h * r
        return x @ self.ent.weight.t()
    
    def forward(self, h_idx, r_idx, t_idx):
        """
        Forward pass for training with negative sampling
        h_idx, r_idx, t_idx: (batch_size,)
        """
        h = self.ent(h_idx)
        r = self.rel(r_idx)
        t = self.ent(t_idx)
        return (h * r * t).sum(dim=1)


class QuestionEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.proj = nn.Linear(768, dim)

    def forward(self, input_ids, attention_mask):
        """
        Encode question to relation-like embedding
        Returns: (batch_size, dim)
        """
        out = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token embedding
        return self.proj(out.last_hidden_state[:, 0])