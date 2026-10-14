"""Validate group-mask fix and measure isolated routing optimization candidates."""
import ast
import inspect
import json
import re
import subprocess
import textwrap
import types
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from torch.distributed.pipelining import microbatch
from torch.nn.attention.flex_attention import BlockMask
microbatch.BlockMask = BlockMask
from nemo_automodel.components.models.glm5_next.model import build_glm5_next_moe_config
from nemo_automodel.components.moe.layers import Gate
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextTopkRouter
from xtuner.v1.module.router import noaux_router as mod

ROOT=Path('/mnt/shared-storage-user/yushan/GLM-5.3-flash-integration')
OUT=Path('/tmp/glm53_topk_fix_validation_20261013')
BASE='df348c2ed'

def forward_source(source):
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NoAuxRouter')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    return textwrap.dedent(ast.get_source_segment(source,method))

def bind(router, source):
    scope=dict(vars(mod));exec(source,scope)
    router.forward=types.MethodType(scope['forward'],router)
    return router

def make(n=288,k=8,groups=1):
    return mod.NoAuxRouter(n_routed_experts=n,num_experts_per_tok=k,router_scaling_factor=2.5,
                          scoring_func='sigmoid',n_group=groups,topk_group=1).cuda()

def main():
    OUT.mkdir(exist_ok=False)
    old=forward_source(subprocess.check_output(['git','show',BASE+':xtuner/v1/module/router/noaux_router.py'],text=True))
    fixed=textwrap.dedent(inspect.getsource(mod.NoAuxRouter.forward))
    unsorted=fixed.replace('torch.topk(selection_scores, k=self.top_k, dim=-1)',
                           'torch.topk(selection_scores, k=self.top_k, dim=-1, sorted=False)')
    # Measurement only: omit a public result field; do not apply this API change to production.
    nohist=re.sub(r'tokens_per_expert = torch.histc\([\s\S]*?\)  # .view\(self.ep_mesh.size\(\), -1\)',
                  'tokens_per_expert = None',unsorted,count=1)
    assert nohist!=unsorted
    report={'base':BASE,'scope':'Same GPU full NoAuxRouter eager forward, excluding gate projection/dispatcher; backward equivalence and group reference checks.'}
    routers={name:bind(make(),src) for name,src in [('old',old),('fixed',fixed),('unsorted',unsorted),('unsorted_nohist_probe_only',nohist)]}
    ckpt=Path('/tmp/GLM-5.3-Flash-4layer');cfg=json.loads((ckpt/'config.json').read_text())['text_config']
    mapping=json.loads((ckpt/'model.safetensors.index.json').read_text())['weight_map']
    def weight(name):
        key='model.language_model.layers.1.mlp.gate.'+name
        with safe_open(str(ckpt/mapping[key]),framework='pt') as f:return f.get_tensor(key).cuda()
    w=weight('weight').bfloat16();bias=weight('e_score_correction_bias').float()
    for r in routers.values():r.e_score_correction_bias.copy_(bias)
    source=Path('/tmp/glm53_xtuner_hf_norm_vs_native_automodel_forward_20261013')
    with safe_open(str(source/'xtuner/hidden_states.safetensors'),framework='pt') as f:
        x=f.get_tensor('layer1.post_attention_layernorm').cuda()[0]
    logits=F.linear(x.float(),w.float()).detach()
    coefficient=torch.linspace(-.7,.9,288,device='cuda')[None]
    outputs={};grads={}
    for name,r in routers.items():
        z=logits.clone().requires_grad_();out=r(z)
        loss=(out['topk_weights']*coefficient.expand(len(z),-1).gather(1,out['topk_ids'])).sum()+.001*(out['router_weights']*coefficient).sum()
        grads[name]=torch.autograd.grad(loss,z)[0]
        order=out['topk_ids'].argsort(-1)
        outputs[name]=(out['topk_ids'].gather(1,order),out['topk_weights'].gather(1,order).detach(),out['router_weights'].detach())
    report['real_input']={}
    for name in routers:
        assert torch.equal(outputs['old'][0],outputs[name][0])
        report['real_input'][name]={'weight_max_abs':float((outputs[name][1]-outputs['old'][1]).abs().max()),
            'gradient_max_abs':float((grads[name]-grads['old']).abs().max()),
            'gradient_relative_l2':float((grads[name]-grads['old']).norm()/grads['old'].norm())}
        assert torch.isfinite(grads[name]).all()
    assert all(torch.equal(a,b) for a,b in zip(outputs['old'],outputs['fixed']))
    assert torch.equal(grads['old'],grads['fixed'])
    with torch.no_grad():
        for r in routers.values():
            for _ in range(20):r(logits)
        timings={name:[] for name in routers}
        for trial in range(9):
            names=list(routers);names=names[trial%4:]+names[:trial%4]
            for name in names:
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100):routers[name](logits)
                end.record();end.synchronize();timings[name].append(start.elapsed_time(end)*10)
    report['forward_timings']={k:{'median_us':sorted(v)[4],'batches_us':v} for k,v in timings.items()}
    # Native reference calls include their own group masking and unbiased weights.
    small=dict(cfg,hidden_size=4,n_routed_experts=4,num_local_experts=4,num_experts_per_tok=2,n_group=2,topk_group=1)
    c=types.SimpleNamespace(**small)
    am=Gate(build_glm5_next_moe_config(c,torch.float32),gate_precision=torch.float32).cuda()
    hf=Glm5NextTextTopkRouter(c).cuda();hf.weight.data.copy_(torch.eye(4,device='cuda'))
    grouped=make(4,2,2);original=bind(make(4,2,2),old)
    b=torch.tensor([-.6,-.7,-.8,-.9],device='cuda')
    for r in (am,hf,grouped,original):r.e_score_correction_bias.copy_(b)
    z=torch.zeros(1,4,device='cuda',requires_grad=True)
    a=grouped(z);before=original(z);aw,ai,_=am._route_scores(z);_,hw,hi=hf(z)
    assert before['topk_ids'].sort(-1).values.tolist()==[[2,3]]
    assert a['topk_ids'].sort(-1).values.tolist()==ai.sort(-1).values.tolist()==hi.sort(-1).values.tolist()==[[0,1]]
    assert torch.isfinite(a['router_weights']).all()
    for name,weights in [('AM',aw),('HF',hw)]:
        torch.testing.assert_close(a['topk_weights'],weights,atol=0,rtol=0)
    report['negative_group_case']={'old':before['topk_ids'].tolist(),'fixed':a['topk_ids'].tolist(),'AM':ai.tolist(),'HF':hi.tolist(),'aux_finite':True}
    # More than one logit value, compare expert-associated weights and Jacobian-vector products.
    torch.manual_seed(42)
    z=(torch.randn(256,4,device='cuda')*.2).requires_grad_()
    def canonical(ids,weights):
        order=ids.argsort(-1);return ids.gather(1,order),weights.gather(1,order)
    a=grouped(z);aw,ai,_=am._route_scores(z);_,hw,hi=hf(z)
    xi,xw=canonical(a['topk_ids'],a['topk_weights'])
    base_loss=(xw*xi.float()).sum();gx=torch.autograd.grad(base_loss,z,retain_graph=True)[0]
    report['random_grouped_reference']={}
    for name,ids,weights in [('AM',ai,aw),('HF',hi,hw)]:
        ids,weights=canonical(ids,weights);assert torch.equal(xi,ids)
        g=torch.autograd.grad((weights*ids.float()).sum(),z,retain_graph=True)[0]
        torch.testing.assert_close(xw,weights,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(gx,g,atol=1e-6,rtol=1e-6)
        report['random_grouped_reference'][name]={'weight_max_abs':float((xw-weights).abs().max()),'gradient_max_abs':float((gx-g).abs().max())}
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
