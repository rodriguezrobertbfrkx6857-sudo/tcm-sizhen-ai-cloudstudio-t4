"""Tiny CPU engineering fixtures, not trained production models."""
import copy,json,tempfile,unittest
from pathlib import Path
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from qiora_rocm.vendor import ProConfig
from qiora_rocm.train import update
from qiora_rocm.training_state import save_boundary,read_owned,verify_state
from qiora_rocm.vendor.checkpoint import load_checkpoint
from qiora_max_rocm.model import QioraMax,LayerRouter
from cloudstudio_tongue.prepare_training import image_row
from cloudstudio_tongue.rocm_train import backup_checkpoint
from PIL import Image

def tiny():
    c=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=64,bos_token_id=1,eos_token_id=2,pad_token_id=0)
    c._attn_implementation='sdpa'
    m=QioraMax(Qwen3ForCausalLM(c),ProConfig(layers=[1],rank=2,alpha=4,dropout=.05,router_hidden=8,bge_dim=4));m.backbone.config.use_cache=False;m.set_phase('R1')
    o=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3)
    return m,o,torch.optim.lr_scheduler.LambdaLR(o,lambda _:1.)

class MaxVisionTests(unittest.TestCase):
    def test_reader_update_and_resume_exact(self):
        torch.set_num_threads(2);torch.manual_seed(17);m,o,s=tiny();before=copy.deepcopy(m.state_dict())
        x=dict(input_ids=[1,4,5,2],labels=[-100,-100,5,2]);update(m,[x],[1.],o,s,'cpu')
        changed=[n for n,v in before.items() if not torch.equal(v,m.state_dict()[n])]
        self.assertTrue(changed);self.assertTrue(all('.reading.' in n for n in changed))
        with tempfile.TemporaryDirectory() as td:
            run=Path(td)/'run';run.mkdir();r=save_boundary(run,Path(td)/'backup',m,o,s,'maxfixture',['a','b'],1,1,4,{})
            update(m,[x],[1.],o,s,'cpu');expected=copy.deepcopy(m.state_dict())
            torch.manual_seed(17);new,no,ns=tiny();cp=read_owned(run/'step_000001.pt',r['checkpoint_sha256'])
            verify_state(dict(cp,phase='T2'),new,'maxfixture',['a','b'],1)
            load_checkpoint(run/'step_000001.pt',new,no,identity='maxfixture',expected_sha256=r['checkpoint_sha256'],scheduler=ns,restore_rng=True)
            update(new,[x],[1.],no,ns,'cpu')
            self.assertTrue(all(torch.equal(v,new.state_dict()[n]) for n,v in expected.items()))
    def test_prompt_layer_router_and_explicit_route(self):
        r=LayerRouter(16,2,8);e=torch.randn(2,4,16,requires_grad=True);mask=torch.ones(2,4)
        w=r(e,mask);self.assertEqual(tuple(w.shape),(2,2,3));self.assertTrue(torch.allclose(w.sum(-1),torch.ones(2,2)))
        (-w[:,:,0].log().mean()).backward();self.assertIsNone(e.grad);self.assertTrue(any(p.grad is not None for p in r.parameters()))
        m,_,_=tiny();m.set_phase('INFERENCE')
        with self.assertRaises(ValueError):m.coefficients(torch.ones(1,2,dtype=torch.long),torch.ones(1,2))
        with self.assertRaises(ValueError):
            with m.routed(torch.tensor([[-1.,0.,0.]])):pass
    def test_image_labels_and_box_bounds(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'images/train').mkdir(parents=True);(root/'labels/train').mkdir(parents=True)
            im=root/'images/train/a.png';Image.new('RGB',(16,16),'red').save(im);lab=root/'labels/train/a.txt';lab.write_text('0 0.5 0.5 0.2 0.2')
            z=image_row(root,im,'train');self.assertEqual(z['boxes'][0][0],0)
            lab.write_text('0 0.99 0.5 0.5 0.2')
            with self.assertRaises(ValueError):image_row(root,im,'train')
    def test_vision_backup_collision_and_readback(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'last.pt';p.write_bytes(b'fixture-not-production');backup=root/'backup';backup.mkdir()
            r=backup_checkpoint(p,backup,0);self.assertEqual(Path(r['path']).read_bytes(),p.read_bytes())
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):backup_checkpoint(p,backup,0)

if __name__=='__main__':unittest.main()
