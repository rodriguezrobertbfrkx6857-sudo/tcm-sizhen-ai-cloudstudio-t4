"""Domain + Evidence + source-reading LoRA, optional prompt-only layer Router."""
from contextvars import ContextVar
from contextlib import contextmanager
from types import MethodType
import re
import torch
from torch import nn
from peft import LoraConfig
from peft.tuners.lora.layer import Linear
from qiora_rocm.vendor import QioraPro

ROUTE=ContextVar('qiora_max_layer_route',default=None)
EXPERTS=('domain','evidence','reading')


def layer_forward(layer,x,*args,**kwargs):
    w=ROUTE.get()
    if w is None or w.shape[0]!=x.shape[0]:raise ValueError('EXPLICIT_MAX_LAYER_ROUTE_REQUIRED')
    if layer.merged or layer.disable_adapters:raise ValueError('NO_MAX_MERGE_OR_DISABLE')
    out=layer.base_layer(x,*args,**kwargs)
    for i,name in enumerate(EXPERTS):
        a,b=layer.lora_A[name],layer.lora_B[name]
        delta=b(a(layer.lora_dropout[name](x.to(a.weight.dtype))))*layer.scaling[name]
        coeff=w[:,layer.max_layer_index,i].reshape(-1,*([1]*(delta.ndim-1))).to(delta)
        out=out+coeff*delta
    return out.to(layer.base_layer.weight.dtype)


class LayerRouter(nn.Module):
    def __init__(self,hidden,layers,width=64):
        super().__init__();self.layers=layers
        self.network=nn.Sequential(nn.Linear(hidden,width),nn.Tanh(),nn.Linear(width,layers*3))
    def forward(self,embedding,mask):
        if (mask.sum(-1)==0).any():raise ValueError('EMPTY_PROMPT')
        m=mask.to(embedding).unsqueeze(-1)
        pooled=(embedding.detach()*m).sum(1)/m.sum(1)
        return self.network(pooled.to(next(self.parameters()).dtype)).reshape(len(mask),self.layers,3).softmax(-1)


class QioraMax(QioraPro):
    def __init__(self,base,config):
        super().__init__(base,config)
        self.backbone.add_adapter('reading',LoraConfig(r=config.rank,lora_alpha=config.alpha,lora_dropout=config.dropout,
            target_modules=self.target_names,bias='none',task_type='CAUSAL_LM',use_dora=False))
        indices=sorted(config.layers);self.max_layers=len(indices)
        for name,layer in self.backbone.named_modules():
            if isinstance(layer,Linear):
                match=re.search(r'layers\.(\d+)\.',name)
                if not match:raise ValueError('MISSING_LAYER_ID')
                layer.max_layer_index=indices.index(int(match[1]));layer.forward=MethodType(layer_forward,layer)
        self.router=LayerRouter(base.config.hidden_size,self.max_layers,config.router_hidden)
        self.learned_router_enabled=False
        self.set_phase('INFERENCE')
    def set_phase(self,phase,**kwargs):
        if phase!='R1':return super().set_phase(phase,**kwargs)
        for n,p in self.named_parameters():
            p.requires_grad_('lora_A.reading.' in n or 'lora_B.reading.' in n);p.grad=None
        self.phase=phase
    def coefficients(self,input_ids,attention_mask,metadata=None,retrieval_embedding=None):
        if self.phase in ('T1','T2','R1'):
            w=torch.zeros(len(input_ids),self.max_layers,3,device=input_ids.device)
            w[:,:,{'T1':0,'T2':1,'R1':2}[self.phase]]=1
            return w
        if not self.learned_router_enabled:raise ValueError('MAX_INFERENCE_REQUIRES_EXPLICIT_FIXED_ROUTE_NO_TRAINED_ROUTER')
        return self.router(self.backbone.get_input_embeddings()(input_ids),attention_mask)
    @contextmanager
    def routed(self,weights):
        if weights.ndim==2:weights=weights[:,None,:].expand(-1,self.max_layers,-1)
        if weights.ndim!=3 or weights.shape[1:]!=(self.max_layers,3) or not torch.isfinite(weights).all() or (weights<0).any():
            raise ValueError('INVALID_MAX_ROUTE')
        with self._lock:
            token=ROUTE.set(weights)
            try:yield
            finally:ROUTE.reset(token)
