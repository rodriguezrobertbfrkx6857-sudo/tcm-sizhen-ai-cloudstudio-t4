"""Disposable tiny CPU optimizer fixtures, never production/medical model results."""
import copy
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from qiora_rocm.vendor import QioraPro,ProConfig
from qiora_rocm.train import update
from qiora_rocm.training_state import save_boundary,restore,read_owned,verify_state,RunLock,require_parent_binding
from qiora_rocm.training_data import encode,rows,isolate,safe_path
from qiora_rocm.dev_eval import verdict,paired_report,fuse


def tiny():
    config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=64,
        bos_token_id=1,eos_token_id=2,pad_token_id=0)
    config._attn_implementation='sdpa'
    model=QioraPro(Qwen3ForCausalLM(config),ProConfig(layers=[1],rank=2,alpha=4,dropout=.05,router_hidden=8,bge_dim=4))
    model.backbone.config.use_cache=False;model.set_phase('T1')
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=0.)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:1.)
    return model,opt,sched


class TrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def setUp(self):random.seed(2);torch.manual_seed(2)
    def test_real_optimizer_resume_exact_and_backup(self):
        model,opt,sched=tiny();initial={n:p.detach().clone() for n,p in model.named_parameters()}
        x=dict(input_ids=[1,4,5,6,7,2],labels=[-100,-100,5,6,7,2]);batch=[x,x];order=['a','b','c','d']
        update(model,batch,[.3,1.],opt,sched,'cpu')
        self.assertTrue(any(not torch.equal(initial[n],p) for n,p in model.named_parameters() if p.requires_grad))
        self.assertTrue(all(torch.equal(initial[n],p) for n,p in model.named_parameters() if not p.requires_grad))
        with tempfile.TemporaryDirectory() as td:
            run=Path(td)/'run';run.mkdir();backup=Path(td)/'backup'
            receipt=save_boundary(run,backup,model,opt,sched,'fixture',order,1,2,12,
                                 dict(parent_manifest_sha='a'*64,parent_id='fixture',code_sha='b'*64,data_sha='c'*64))
            update(model,batch,[.3,1.],opt,sched,'cpu');expected=copy.deepcopy(model.state_dict())
            # Adapter-only resume requires the identical frozen parent, not another random base.
            torch.manual_seed(2)
            new,o,s=tiny();restore(run/'step_000001.pt',receipt['checkpoint_sha256'],new,o,s,'fixture',order,2)
            update(new,batch,[.3,1.],o,s,'cpu')
            self.assertTrue(all(torch.equal(v,new.state_dict()[n]) for n,v in expected.items()))
            self.assertEqual(o.state_dict()['param_groups'],opt.state_dict()['param_groups'])
            self.assertEqual(s.state_dict(),sched.state_dict())
            self.assertEqual(hashlib.sha256((backup/'run/step_000001.pt').read_bytes()).hexdigest(),receipt['checkpoint_sha256'])
            cp=read_owned(run/'step_000001.pt',receipt['checkpoint_sha256'])
            require_parent_binding(cp,'a'*64)
            with self.assertRaises(ValueError):require_parent_binding(cp,'d'*64)
            for field,value in [('identity','bad'),('offset',3),('data_order',['x']),('optimizer',{}),('scheduler',None),('torch_rng',None)]:
                bad=copy.deepcopy(cp);bad[field]=value
                with self.assertRaises(ValueError):verify_state(bad,new,'fixture',order,2)
            with self.assertRaises(ValueError):read_owned(run/'step_000001.pt','0'*64)
            with self.assertRaises(ValueError):save_boundary(run,backup,new,o,s,'fixture',order,1,2,12,{})
    def test_evidence_only_real_update_and_fusion(self):
        m,_,_=tiny();m.set_phase('T2');before=copy.deepcopy(m.state_dict())
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4);s=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:1.)
        update(m,[dict(input_ids=[1,4,5,2],labels=[-100,-100,5,2])],[1.],opt,s,'cpu')
        changed=[n for n,v in before.items() if not torch.equal(v,m.state_dict()[n])]
        self.assertTrue(changed);self.assertTrue(all('.evidence.' in n for n in changed))
        cp=dict(config=m.pro_config.to_dict(),phase='T2',adapter_router_state=m.state_dict())
        new,_,_=tiny();fuse(new,evidence=cp)
        self.assertTrue(all(torch.equal(v,new.state_dict()[n]) for n,v in m.state_dict().items() if '.evidence.' in n))
        cp['phase']='T1'
        with self.assertRaises(ValueError):fuse(new,evidence=cp)
    def test_single_controller_lock(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'lock'
            with RunLock(path):
                with self.assertRaises(FileExistsError):
                    with RunLock(path):pass
            self.assertFalse(path.exists())
    def test_no_truncation_and_mask(self):
        class Tok:
            eos_token_id=2
            def apply_chat_template(self,*a,**kw):return 'prompt'
            def encode(self,text,**kw):return [3,4,5] if text=='prompt' else [6,7]
        x=encode(Tok(),dict(question='q',answer='a'),6)
        self.assertEqual(x['labels'],[-100,-100,-100,6,7,2])
        with self.assertRaises(ValueError):encode(Tok(),dict(question='q',answer='a'),5)
        with self.assertRaises(ValueError):encode(Tok(),dict(question='q',answer='<|im_end|>'),32)
    def test_split_sha_and_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'fixture.jsonl';r=dict(sample_id='a',question='q',answer='a',fact_cluster_id='f',quality_tier='FIXTURE',source_type='FIXTURE',work_id='w',train_role='TRAIN',split='TRAIN')
            p.write_text(json.dumps(r)+'\n');h=hashlib.sha256(p.read_bytes()).hexdigest()
            rr=rows(p,h,'TRAIN');self.assertEqual(len(rr),1)
            with self.assertRaises(ValueError):rows(p,'0'*64,'TRAIN')
            with self.assertRaises(ValueError):rows(p,h,'DEV')
            with self.assertRaises(ValueError):isolate(rr,[dict(fact_cluster_id='f')])
            with self.assertRaises(ValueError):safe_path(Path(td)/'TEST2.jsonl')
    def test_raw_score_unknown_and_paired_regression(self):
        item=dict(sample_id='a',fact_cluster_id='f',kind='EXACT_NONMEDICAL',expected_answer='5')
        panel=[dict(sample_id='a',route='base',raw='5',condition_sha='x',truncated=False),
               dict(sample_id='a',route='domain',raw='6',condition_sha='x',truncated=False)]
        z=paired_report([item],panel);self.assertFalse(z['PRO_V1_INITIAL_ACCEPTED']);self.assertEqual(len(z['confirmed_new_errors']),1)
        self.assertEqual(z['summary']['domain']['dimensions']['GENERAL']['absolute_gain'],-1.)
        self.assertEqual(verdict(dict(task_type='missing'),'无法确认')['state'],'NEEDS_REVIEW')
        panel[1]['condition_sha']='different'
        with self.assertRaises(ValueError):paired_report([item],panel)
    def test_compact_relation_no_medical_claim(self):
        item=dict(contract_version='KER_COMPACT_RELATION_V3',claim=dict(kind='literal'),sources=[])
        self.assertEqual(verdict(item,'{"v":"KER_COMPACT_RELATION_V3","relation":"MISSING","evidence_ids":[]}')['state'],'CORRECT')
        self.assertEqual(verdict(item,'{"v":"KER_COMPACT_RELATION_V3","relation":"SUPPORTED","evidence_ids":[]}')['state'],'INCORRECT')
        self.assertEqual(verdict(item,'not json')['state'],'NEEDS_REVIEW')


if __name__=='__main__':unittest.main()
