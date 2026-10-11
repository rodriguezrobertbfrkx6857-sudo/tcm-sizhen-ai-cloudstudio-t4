import torch
from torch import nn

class SequenceRouter(nn.Module):
    def __init__(self, feature_dim, hidden=64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(feature_dim, hidden), nn.Tanh(), nn.Linear(hidden, 3))

    def forward(self, features, active_mask):
        mask = torch.as_tensor(active_mask, dtype=torch.bool, device=features.device)
        if mask.shape != (3,) or not mask.any():
            raise ValueError('Router requires at least one active expert')
        logits = self.network(features)
        return logits.masked_fill(~mask, -torch.inf).softmax(-1)

def utilization(weights, intents):
    w = weights.detach().float().cpu()
    if not len(w):
        raise ValueError('Cannot report empty routing measurements')
    entropy = -(w * w.clamp_min(1e-12).log()).sum(-1)
    per_intent = {i: w[[j for j, s in enumerate(intents) if s == i]].mean(0).tolist() for i in sorted(set(intents))}
    dominant = (w.max(-1).values > .9).float().mean().item()
    return {'samples': len(w), 'per_expert_mean': w.mean(0).tolist(), 'per_intent_mean': per_intent,
            'entropy_mean': entropy.mean().item(), 'expert_utilization_gt_0_1': (w > .1).float().mean(0).tolist(),
            'fraction_dominant_gt_0_9': dominant, 'active_distribution': (w > 0).float().mean(0).tolist(),
            'collapse_warning': len(w) >= 32 and dominant > .9,
            'interpretation': 'Routing coefficients only; not diagnosis or source-truth confidence'}
