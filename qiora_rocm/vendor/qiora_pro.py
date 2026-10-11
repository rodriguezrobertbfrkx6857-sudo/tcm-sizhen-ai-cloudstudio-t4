"""PEFT tensors with one set of sequence coefficients shared by all chosen layers.

No attention/backbone pass is used to obtain coefficients. Context must span
backward when checkpoint recomputation is used; production trainer uses no
gradient checkpointing until separately validated.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from types import MethodType
import threading
import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Linear
from .config import EXPERTS, case_enabled
from .router.features import FeatureExtractor
from .router.sequence_router import SequenceRouter

_ROUTE = ContextVar('qiora_route', default=None)

def weighted_lora_forward(layer, x, *args, **kwargs):
    weights = _ROUTE.get()
    if weights is None:
        raise RuntimeError('Qiora expert forward requires an explicit routed context')
    if layer.merged or layer.disable_adapters:
        raise RuntimeError('Dynamic expert layers cannot be statically merged/disabled')
    result = layer.base_layer(x, *args, **kwargs)
    dtype = result.dtype
    if x.shape[0] != weights.shape[0]:
        raise ValueError('Sequence batch mismatch (beam expansion is not supported)')
    for i, expert in enumerate(EXPERTS):
        if expert not in layer.lora_A:
            continue
        A, B = layer.lora_A[expert], layer.lora_B[expert]
        delta = B(A(layer.lora_dropout[expert](x.to(A.weight.dtype)))) * layer.scaling[expert]
        coeff = weights[:, i].view(-1, *([1] * (delta.ndim - 1))).to(delta)
        result = result + coeff * delta
    return result.to(dtype)

class QioraPro(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        self.pro_config = config
        self.target_names = config.discover(backbone)
        adapter_config = LoraConfig(r=config.rank, lora_alpha=config.alpha, lora_dropout=config.dropout,
                                    target_modules=self.target_names, bias='none', task_type='CAUSAL_LM', use_dora=False)
        self.backbone = get_peft_model(backbone, adapter_config, adapter_name='domain')
        self.backbone.add_adapter('evidence', adapter_config)
        # Case is represented even while inactive; random case tensors never receive weight.
        self.backbone.add_adapter('case', adapter_config)
        for m in self.backbone.modules():
            if isinstance(m, Linear):
                m.forward = MethodType(weighted_lora_forward, m)
        hidden = backbone.config.hidden_size
        self.features = FeatureExtractor(hidden, config.bge_dim)
        self.router = SequenceRouter(self.features.output_dim, config.router_hidden)
        # Even a capability config cannot activate randomly initialized Case tensors.
        # A verified checkpoint must be loaded before explicit activate_case().
        self.active_mask = (True, True, False)
        self._lock = threading.RLock()
        self.phase = 'INFERENCE'
        self.set_phase('INFERENCE')

    def set_phase(self, phase, *, case_data_pass=False):
        if phase not in ('T1', 'T2', 'T3', 'T5', 'INFERENCE'):
            raise ValueError('Unsupported production phase')
        if phase == 'T5' and not case_data_pass:
            raise ValueError('Case training requires a separate eligible-data gate')
        for p in self.parameters():
            p.requires_grad_(False)
            p.grad = None
        expert = {'T1': 'domain', 'T2': 'evidence', 'T5': 'case'}.get(phase)
        for name, p in self.named_parameters():
            if expert and (f'lora_A.{expert}.' in name or f'lora_B.{expert}.' in name):
                p.requires_grad_(True)
            elif phase == 'T3' and name.startswith('router.'):
                p.requires_grad_(True)
        self.phase = phase

    def activate_case(self, capability):
        # Post-evaluation deployment capability is not a mutation of training hyperparameters.
        if self.phase != 'INFERENCE' or not case_enabled(capability):
            raise ValueError('Case production activation lacks trained/evaluated capability')
        if getattr(self, 'loaded_checkpoint_sha256', None) != capability.get('checkpoint_sha256'):
            raise ValueError('Case proof is for a different loaded checkpoint')
        self.active_mask = (True, True, True)

    def coefficients(self, input_ids, attention_mask, metadata, retrieval_embedding=None):
        if self.phase in ('T1', 'T2', 'T5'):
            w = torch.zeros(len(input_ids), 3, device=input_ids.device)
            w[:, {'T1': 0, 'T2': 1, 'T5': 2}[self.phase]] = 1
            return w
        embeddings = self.backbone.get_input_embeddings()(input_ids)
        f = self.features(embeddings, attention_mask, metadata, retrieval_embedding)
        return self.router(f.to(next(self.router.parameters()).dtype), self.active_mask)

    @contextmanager
    def routed(self, weights):
        with self._lock:
            token = _ROUTE.set(weights)
            try:
                yield
            finally:
                _ROUTE.reset(token)

    def forward(self, input_ids, attention_mask=None, metadata=None, retrieval_embedding=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if metadata is None:
            metadata = [{} for _ in input_ids]
        weights = self.coefficients(input_ids, attention_mask, metadata, retrieval_embedding)
        with self.routed(weights):
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return output, weights

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, metadata, **kwargs):
        if kwargs.get('num_beams', 1) != 1 or kwargs.get('num_return_sequences', 1) != 1:
            raise ValueError('Only single-beam sequence routing is validated')
        weights = self.coefficients(input_ids, attention_mask, metadata)
        with self.routed(weights):
            return self.backbone.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def parameter_audit(self):
        names = {n: p.numel() for n, p in self.named_parameters() if p.requires_grad}
        return {'phase': self.phase, 'trainable_parameters': sum(names.values()), 'whitelist': names,
                'total_parameters': sum(p.numel() for p in self.parameters()), 'targets': self.target_names,
                'active_mask': list(self.active_mask)}
