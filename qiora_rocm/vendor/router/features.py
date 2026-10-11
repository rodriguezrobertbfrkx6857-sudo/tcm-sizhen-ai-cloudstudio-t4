"""Embedding lookup only, not a transformer forward or a new retrieval call."""
import torch
from torch import nn

INTENTS = ('general', 'fact', 'comparison', 'explanation', 'case')
SOURCES = ('none', 'book', 'formula', 'case', 'mixed')
MODES = ('normal', 'reasoning', 'grounded')

class FeatureExtractor(nn.Module):
    def __init__(self, embedding_dim, bge_dim=512):
        super().__init__()
        self.embedding_dim, self.bge_dim = embedding_dim, bge_dim
        self.output_dim = embedding_dim + bge_dim + len(INTENTS) + len(SOURCES) + len(MODES) + 4

    def forward(self, embeddings, attention_mask, metadata, retrieval_embedding=None):
        batch = embeddings.shape[0]
        if len(metadata) != batch:
            raise ValueError('One metadata item per sequence')
        mask = attention_mask.to(embeddings.device).float().unsqueeze(-1)
        pooled = (embeddings.detach().float() * mask).sum(1) / mask.sum(1).clamp_min(1)
        bge = torch.zeros(batch, self.bge_dim, device=embeddings.device)
        if retrieval_embedding is not None:
            if retrieval_embedding.shape != bge.shape:
                raise ValueError('Invalid precomputed BGE shape')
            if any(m.get('mode', 'normal') != 'grounded' for m in metadata):
                raise ValueError('BGE reuse is grounded-only; no retrieval in normal routing')
            bge = retrieval_embedding.detach().to(bge)
        values = []
        for m in metadata:
            v = []
            for key, categories, default in [('intent', INTENTS, 'general'),
                                              ('source_type', SOURCES, 'none'),
                                              ('mode', MODES, 'normal')]:
                value = m.get(key, default)
                if value not in categories:
                    raise ValueError('Unknown router metadata: ' + key)
                v.extend(float(value == c) for c in categories)
            confidence = float(m.get('retrieval_confidence', 0))
            if not 0 <= confidence <= 1:
                raise ValueError('Retrieval confidence must be bounded')
            v.extend([confidence, float(bool(m.get('evidence_available', False))),
                      float(bool(m.get('structured_query', False))), float(retrieval_embedding is not None)])
            values.append(v)
        return torch.cat([pooled, bge, torch.tensor(values, device=pooled.device)], dim=1)
