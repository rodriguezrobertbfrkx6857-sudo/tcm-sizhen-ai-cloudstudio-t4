from dataclasses import dataclass, field, asdict
import re

EXPERTS = ('domain', 'evidence', 'case')

@dataclass
class ProConfig:
    placement: str = 'MID_LATE'
    layers: list[int] | None = None
    targets: tuple[str, ...] = ('q_proj', 'v_proj')
    rank: int = 8
    alpha: int = 16
    dropout: float = .05
    router_hidden: int = 64
    bge_dim: int = 512
    case_capability: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def discover(self, backbone):
        count = backbone.config.num_hidden_layers
        candidates = {'MID': range(count // 3, 2 * count // 3),
                      'MID_LATE': range(count // 2, count),
                      'LATE': range(2 * count // 3, count)}
        if self.placement not in candidates or self.rank < 1:
            raise ValueError('Invalid placement/rank')
        indices = set(self.layers if self.layers is not None else candidates[self.placement])
        if not indices or any(i < 0 or i >= count for i in indices):
            raise ValueError('Invalid inspected layer indices')
        names = []
        for name, module in backbone.named_modules():
            match = re.search(r'(?:^|\.)layers\.(\d+)\.', name)
            if match and int(match.group(1)) in indices and name.rsplit('.', 1)[-1] in self.targets:
                if not hasattr(module, 'in_features') or not hasattr(module, 'out_features'):
                    raise ValueError('Target is not linear: ' + name)
                names.append(name)
        if len(names) != len(indices) * len(self.targets):
            raise ValueError('Module discovery incomplete; do not guess target names')
        return names

def case_enabled(capability):
    from .capability import proof_passes
    required = ('schema_pass', 'provenance_pass', 'privacy_pass', 'rights_pass',
                'training_complete', 'independent_eval_pass')
    return all(capability.get(k) is True for k in required) and proof_passes(capability,'case')
