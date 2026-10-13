"""Audit native routing: group exclusion, near-boundary sensitivity, and eager top-k cost."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
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
from xtuner.v1.module.router.noaux_router import NoAuxRouter

ROOT = Path('/mnt/shared-storage-user/yushan/GLM-5.3-flash-integration')
OUT = Path('/tmp/glm53_topk_review_20261013')

def main():
    OUT.mkdir(exist_ok=False)
    ckpt = Path('/tmp/GLM-5.3-Flash-4layer')
    cfg = json.loads((ckpt/'config.json').read_text())['text_config']
    cfg['num_local_experts'] = cfg['n_routed_experts']
    mapping = json.loads((ckpt/'model.safetensors.index.json').read_text())['weight_map']
    def weight(name):
        key='model.language_model.layers.1.mlp.gate.'+name
        with safe_open(str(ckpt/mapping[key]),framework='pt') as f:
            return f.get_tensor(key).cuda()
    w=weight('weight').bfloat16(); bias=weight('e_score_correction_bias').float()
    valid=~load_file(str(ROOT/'validation/train_infra/forward_alignment_20261013/pack_rank0_step0.safetensors'))['padding_mask'][0].cuda().bool()
    report={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'scope':'Eager single H200, native router group counterexample and saved layer1 input score margins. Timings exclude gate GEMM, dispatcher and backward.'}
    for label,source in [('native','/tmp/glm53_xtuner_hf_norm_vs_native_automodel_forward_20261013'),('shared','/tmp/glm53_xtuner_automodel_shared_mlp_v2_forward_20261013')]:
        scores=[]
        for side in ('xtuner','automodel'):
            with safe_open(str(Path(source)/side/'hidden_states.safetensors'),framework='pt') as f:
                x=f.get_tensor('layer1.post_attention_layernorm').cuda()[0]
            scores.append(F.linear(x.float(),w.float()).sigmoid()+bias)
        a,b=scores
        ranked,ids=a.topk(9,dim=-1)
        margin=ranked[:,7]-ranked[:,8]
        delta=(a-b).abs().amax(-1)
        ix=a.topk(8,dim=-1).indices
        iy=b.topk(8,dim=-1).indices
        changed=(ix.sort(-1).values!=iy.sort(-1).values).any(-1)
        assert not (changed & (margin>2*delta) & valid).any()
        unsorted=a.topk(8,dim=-1,sorted=False).indices
        assert torch.equal(ix.sort(-1).values,unsorted.sort(-1).values)
        bf16=((a-bias).bfloat16().float()+bias).topk(8,dim=-1).indices
        qchange=(ix.sort(-1).values!=bf16.sort(-1).values).any(-1)
        report[label]={'changed_tokens':int(changed[valid].sum()),'exact_8_9_ties':int((margin[valid]==0).sum()),
            'boundary_margin_quantiles':torch.quantile(margin[valid],torch.tensor([0.,.1,.5,.9,1.],device='cuda')).tolist(),
            'changed_margin_median':float(margin[changed & valid].median()),
            'stable_certified_margin_gt_2delta':int(((margin>2*delta)&valid).sum()),
            'bf16_sigmoid_rounding_changed_tokens':int(qchange[valid].sum()),
            'sorted_unsorted_same_expert_sets':True}
        if label=='native':
            def measure(fn):
                for _ in range(10): fn()
                torch.cuda.synchronize()
                samples=[]
                for _ in range(5):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(50): fn()
                    end.record(); end.synchronize(); samples.append(start.elapsed_time(end)*1000/50)
                return {'median_us':sorted(samples)[2],'five_batch_us':samples}
            report['timings']={name:measure(fn) for name,fn in {
                'topk_sorted':lambda:a.topk(8,dim=-1,sorted=True),
                'topk_unsorted':lambda:a.topk(8,dim=-1,sorted=False),
                'unused_histc_with_cast':lambda:torch.histc(ix.float(),bins=288,min=0,max=288),
            }.items()}
    # Every chosen-group score is negative: zero-masking incorrectly promotes excluded experts.
    small=dict(cfg, hidden_size=4,n_routed_experts=4,num_local_experts=4,num_experts_per_tok=2,n_group=2,topk_group=1)
    c=SimpleNamespace(**small)
    xt=NoAuxRouter(n_routed_experts=4,num_experts_per_tok=2,router_scaling_factor=c.routed_scaling_factor,
        scoring_func='sigmoid',n_group=2,topk_group=1).cuda()
    am=Gate(build_glm5_next_moe_config(c,torch.float32),gate_precision=torch.float32).cuda()
    hf=Glm5NextTextTopkRouter(c).cuda()
    bias=torch.tensor([-.6,-.7,-.8,-.9],device='cuda')
    for module in (xt,am,hf): module.e_score_correction_bias.copy_(bias)
    am.weight.zero_();hf.weight.zero_()
    logits=torch.zeros(1,4,device='cuda')
    xi=xt(logits)['topk_ids'];_,ai,_=am._route_scores(logits);_,_,hi=hf(logits)
    assert xi.sort(-1).values.tolist()==[[2,3]]
    assert ai.sort(-1).values.tolist()==hi.sort(-1).values.tolist()==[[0,1]]
    report['negative_group_mask_counterexample']={'bias':bias.tolist(),'XT_ids':xi.tolist(),'AM_ids':ai.tolist(),'HF_ids':hi.tolist(),
        'affected_current_experiment':False,'reason':'Current GLM uses n_group=topk_group=1.'}
    # A naive 0 -> -inf edit would leak -inf into XT auxiliary normalization.
    masked=torch.tensor([[-.1,-.2,float('-inf'),float('-inf')]],device='cuda')
    report['naive_inf_patch_aux_nonfinite']=not bool(torch.isfinite(masked/masked.sum(-1,keepdim=True)).all())
    assert report['naive_inf_patch_aux_nonfinite']
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    with torch.no_grad(): main()
