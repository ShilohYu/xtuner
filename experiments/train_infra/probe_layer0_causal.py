"""Replay native layer0, locate first divergence, and ablate one precision boundary at a time."""
import json
import subprocess
from pathlib import Path
from types import MethodType, SimpleNamespace
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch.distributed.pipelining import microbatch
from torch.nn.attention.flex_attention import BlockMask
microbatch.BlockMask=BlockMask
from nemo_automodel.components.models.glm5_next import layers as am_layers
from nemo_automodel.components.models.glm5_next.config import Glm5NextTextConfig
from nemo_automodel.components.models.glm5_next.model import build_glm5_next_moe_config,_packed_context_from_inputs
from nemo_automodel.components.models.common import BackendConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextRMSNormGated,Glm5NextTextMLP,Glm5NextTextHyperConnection
from xtuner.v1.model.moe.glm53 import Glm53TextMoEConfig
from xtuner.v1.model.moe.glm53 import decoder_layer as xt_layers
from xtuner.v1.data_proto import SequenceContext
from experiments.train_infra.text_attention import install_xtuner_hf_rms_norm
from experiments.train_infra.plot_forward import errors

ROOT=Path('/mnt/shared-storage-user/yushan/GLM-5.3-flash-integration')
SOURCE=Path('/tmp/glm53_xtuner_hf_norm_vs_native_automodel_forward_20261013')
OUT=Path('/tmp/glm53_layer0_causal_v3_20261013')

def main():
    OUT.mkdir(exist_ok=False)
    checkpoint=Path('/tmp/GLM-5.3-Flash-4layer')
    config=json.loads((checkpoint/'config.json').read_text())['text_config']
    cfg=Glm5NextTextConfig(**config);xcfg=Glm53TextMoEConfig.from_hf(checkpoint)
    mapping=json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    def weight(name):
        key='model.language_model.layers.0.'+name
        with safe_open(str(checkpoint/mapping[key]),framework='pt') as f:return f.get_tensor(key).cuda()
    def capture(side,name):
        with safe_open(str(SOURCE/side/'hidden_states.safetensors'),framework='pt') as f:return f.get_tensor('layer0'+name).cuda()
    xt=xt_layers.Glm53DenseDecoderLayer(hidden_size=xcfg.hidden_size,intermediate_size=xcfg.intermediate_size,
        hidden_act=xcfg.hidden_act,swiglu_limit=xcfg.swiglu_limit,rms_norm_eps=xcfg.rms_norm_eps,
        attention_config=xcfg.linear_attention,layer_idx=0,mhc_cfg=xcfg.mhc).cuda().bfloat16()
    am=am_layers.Glm5NextDecoderLayer(cfg,0,build_glm5_next_moe_config(cfg,torch.bfloat16),BackendConfig(attn='cudnn',linear='torch')).cuda()
    for side,model in [('xtuner',xt),('automodel',am)]:
        for name,p in model.named_parameters():
            key=name.replace('self_attn._fp32_params.','self_attn.')
            for site in ('attn','ffn'):
                key=key.replace(site+'_hc._fp32_params.','hc_'+site+'_').replace(site+'_hc.','hc_'+site+'_')
            p.data=weight(key).to(torch.float32 if any(v in key for v in ['_base','_scale','A_log','dt_bias','conv1d.weight']) else torch.bfloat16)
        model.train()
    install_xtuner_hf_rms_norm(xt)
    fixture={k:v.cuda() for k,v in load_file(str(ROOT/'validation/train_infra/forward_alignment_20261013/pack_rank0_step0.safetensors')).items()}
    valid=~fixture['padding_mask'][0].bool()
    context=SequenceContext(input_ids=fixture['input_ids'],cu_seq_lens_q=fixture['cu_seqlens'],cu_seq_lens_k=fixture['cu_seqlens'],device='cuda',max_length_q=int(fixture['cu_seqlens'].diff().max()),max_length_k=int(fixture['cu_seqlens'].diff().max()))
    packed=_packed_context_from_inputs(fixture['input_ids'],None,fixture['padding_mask'],fixture['cu_seqlens'],None)
    x=capture('xtuner','.input');assert torch.equal(x,capture('automodel','.input'))
    report={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'scope':'Native standalone layer0, shared recorded 8192-token residual input and checkpoint, train/no_grad, FP32 holders incl conv weights. HF-norm experiment retained on XT. Real-token metrics exclude padding.'}
    stages={};handles=[]
    def hook(name):
        def cb(module,args,output):
            value=output['projected_output'] if isinstance(output,dict) else output[0] if isinstance(output,tuple) else output
            stages[name]=value.detach()
            if name.endswith('o_norm'):
                stages[name+'.input']=args[0].detach();stages[name+'.gate']=args[1].detach()
        return cb
    for side,model in [('XT',xt),('AM',am)]:
        for name in ['input_layernorm','self_attn','post_attention_layernorm','mlp','self_attn.q_proj','self_attn.q_conv1d','self_attn.k_conv1d','self_attn.v_conv1d','self_attn.o_norm','self_attn.o_proj']:
            handles.append(model.get_submodule(name).register_forward_hook(hook(side+'.'+name)))
    def run(side):
        return xt._forward(x,position_embeddings=None,seq_ctx=context) if side=='XT' else am(x,packed_context=packed,padding_mask=fixture['padding_mask'])
    def compare(name,a,b):
        r=errors(a[:,valid],b[:,valid]);report[name]=r;print(name,r,flush=True)
    xo,ao=run('XT'),run('AM')
    compare('baseline.layer_output',xo,ao)
    for name in ['input_layernorm','self_attn','post_attention_layernorm','mlp','self_attn.q_proj','self_attn.q_conv1d','self_attn.k_conv1d','self_attn.v_conv1d','self_attn.o_norm.input','self_attn.o_norm.gate','self_attn.o_norm']:
        compare('baseline.'+name,stages['XT.'+name],stages['AM.'+name])
    for side,value in [('xtuner',xo),('automodel',ao)]:
        compare(side+'.runtime_replay',capture(side,''),value)
        assert torch.equal(capture(side,'')[:,valid],value[:,valid]),side+' replay differs'
    baseline_norm_input=stages['XT.self_attn.o_norm.input'];baseline_gate=stages['XT.self_attn.o_norm.gate']
    hf_norm=Glm5NextTextRMSNormGated(cfg.linear_head_dim,eps=cfg.rms_norm_eps).cuda().bfloat16()
    hf_norm.weight.copy_(xt.self_attn.o_norm.weight)
    hn=hf_norm(baseline_norm_input,baseline_gate)
    compare('same_input.gated_norm_XT_vs_HF',xt.self_attn.o_norm(baseline_norm_input,baseline_gate),hn)
    compare('same_input.gated_norm_AM_vs_HF',am.self_attn.o_norm(baseline_norm_input,baseline_gate),hn)
    # Isolate attention from upstream mHC perturbation, then change only AM gated norm.
    same_input=stages['XT.input_layernorm']
    from xtuner.v1.module.attention import kda as xt_kda
    from fla.ops.kda import chunk_kda as fla_chunk_kda
    raw={}
    native_xt_chunk=xt_kda.chunk_kda
    native_am_chunk=am_layers._chunk_kda
    def trace_xt(**kwargs):
        raw['XT']={k:v for k,v in kwargs.items() if isinstance(v,torch.Tensor)}
        out=native_xt_chunk(**kwargs);raw['XT']['out']=out[0];return out
    def trace_am(**kwargs):
        raw['AM']={k:v for k,v in kwargs.items() if isinstance(v,torch.Tensor)}
        out=native_am_chunk(**kwargs);raw['AM']['out']=out[0];return out
    xt_kda.chunk_kda=trace_xt;am_layers._chunk_kda=trace_am
    xx=xt.self_attn(same_input,seq_ctx=context)['projected_output']
    aa=am.self_attn(same_input,packed_context=packed,padding_mask=fixture['padding_mask'])
    compare('same_input.KDA_native',xx,aa)
    compare('same_input.KDA_core',stages['XT.self_attn.o_norm.input'],stages['AM.self_attn.o_norm.input'])
    for key in ('q','k','v','g','beta'):
        compare('same_input.core_input.'+key,raw['XT'][key],raw['AM'][key])
        assert torch.equal(raw['XT'][key],raw['AM'][key]),key
    xt_kda.chunk_kda=native_xt_chunk;am_layers._chunk_kda=native_am_chunk
    # Independent FP32 recurrence over the first 128 positions of the first document.
    # Match the kernels' BF16 normalized q/k inputs, but perform state updates in FP32.
    q,k,v,g,beta=(raw['XT'][key][:,:128] for key in ('q','k','v','g','beta'))
    def normalize(t):
        return (t.float()*torch.rsqrt(t.float().square().sum(-1,keepdim=True)+1e-6)).to(t.dtype).float()
    q,k=normalize(q),normalize(k)
    state=torch.zeros((1,64,128,128),device='cuda');oracle=[]
    for token in range(128):
        state=state*g[:,token].float().exp().unsqueeze(-1)
        kt=k[:,token];vt=v[:,token].float()
        prediction=torch.einsum('bhd,bhdv->bhv',kt,state)
        error=(vt-prediction)*beta[:,token].float().unsqueeze(-1)
        state=state+torch.einsum('bhd,bhv->bhdv',kt,error)
        oracle.append(torch.einsum('bhd,bhdv->bhv',q[:,token]*(128**-.5),state))
    oracle=torch.stack(oracle,dim=1)
    report['KDA_FP32_recurrence_first128']={side:errors(oracle,raw[side]['out'][:,:128]) for side in ('XT','AM')}
    print('KDA_FP32_recurrence_first128',report['KDA_FP32_recurrence_first128'],flush=True)
    del raw,state,oracle,q,k,v,g,beta
    native_am_norm=am.self_attn.o_norm
    am.self_attn.o_norm=hf_norm
    compare('same_input.KDA_AM_HF_gated_norm',xx,am.self_attn(same_input,packed_context=packed,padding_mask=fixture['padding_mask']))
    compare('only_AM_HF_gated_norm.layer_output',xo,run('AM'))
    # Each further intervention is HF-anchored; XT post expression is a diagnostic alignment mode.
    hf_mlp=Glm5NextTextMLP(SimpleNamespace(**config)).cuda().bfloat16()
    hf_mlp.load_state_dict(am.mlp.state_dict(),strict=True)
    hf_mlp.register_forward_hook(hook('AM.mlp'))
    am.self_attn.o_norm=native_am_norm
    native_am_mlp=am.mlp;am.mlp=hf_mlp
    compare('only_AM_HF_MLP.layer_output',xo,run('AM'))
    am.mlp=native_am_mlp
    hf_pre=Glm5NextTextHyperConnection(cfg).cuda()
    native_xt_pre=xt_layers.hc_pre
    def reference_pre(states,fn,scale,base,*args,**kwargs):
        hf_pre.fn=torch.nn.Parameter(fn);hf_pre.base=torch.nn.Parameter(base);hf_pre.scale=torch.nn.Parameter(scale)
        post,comb,collapsed=hf_pre(states)
        return collapsed,post,comb
    xt_layers.hc_pre=reference_pre
    compare('only_XT_HF_hc_pre.layer_output',run('XT'),ao)
    xt_layers.hc_pre=native_xt_pre
    native_xt_post=xt_layers.hc_post
    def reference_post(update,residual,post,comb):
        return post.to(update.dtype).unsqueeze(-1)*update.unsqueeze(-2)+torch.matmul(comb.to(update.dtype).transpose(-1,-2),residual)
    xt_layers.hc_post=reference_post
    compare('only_XT_HF_hc_post.layer_output',run('XT'),ao)
    xt_layers.hc_post=native_xt_post
    am.self_attn.o_norm=hf_norm;am.mlp=hf_mlp
    compare('combined_AM_HF_norm_MLP.layer_output',xo,run('AM'))
    xt_layers.hc_pre=reference_pre
    compare('combined_plus_XT_HF_hc_pre.layer_output',run('XT'),run('AM'))
    xt_layers.hc_post=reference_post
    xf,af=run('XT'),run('AM')
    compare('combined_plus_XT_HF_hc_post.layer_output',xf,af)
    for name in ['input_layernorm','self_attn','post_attention_layernorm','mlp']:
        compare('combined.'+name,stages['XT.'+name],stages['AM.'+name])
    # Replace only the prefix scan to test the native-core difference causally.
    import importlib
    from fla.ops.utils import chunk_local_cumsum
    chunk_module=importlib.import_module('xtuner.v1.ops.kda.chunk_kda')
    native_scan=chunk_module.chunk_cumsum
    def reference_scan(g,cu_seqlens,chunk_indices,scale):
        return chunk_local_cumsum(g=g,chunk_size=64,scale=scale,cu_seqlens=cu_seqlens,chunk_indices=chunk_indices)
    chunk_module.chunk_cumsum=reference_scan
    xp,ap=run('XT'),run('AM')
    compare('only_scan_replaced_on_combined.core_output',stages['XT.self_attn.o_norm.input'],stages['AM.self_attn.o_norm.input'])
    compare('only_scan_replaced_on_combined.layer_output',xp,ap)
    assert torch.equal(stages['XT.self_attn.o_norm.input'][:,valid],stages['AM.self_attn.o_norm.input'][:,valid])
    chunk_module.chunk_cumsum=native_scan
    # Use the existing FLA reference backend for XT's KDA only; retain native AM kernels.
    xt_kda.chunk_kda=fla_chunk_kda
    xf,af=run('XT'),run('AM')
    compare('combined_plus_XT_FLA_core.layer_output',xf,af)
    compare('combined_plus_XT_FLA_core.core_output',stages['XT.self_attn.o_norm.input'],stages['AM.self_attn.o_norm.input'])
    xt.self_attn.o_norm=hf_norm
    xf,af=run('XT'),run('AM')
    compare('HF_reference_aligned.layer_output',xf,af)
    for name in ['input_layernorm','self_attn','post_attention_layernorm','mlp']:
        compare('HF_reference_aligned.'+name,stages['XT.'+name],stages['AM.'+name])
        assert torch.equal(stages['XT.'+name][:,valid],stages['AM.'+name][:,valid]),name
    assert torch.equal(xf[:,valid],af[:,valid]),'Layer0 reference alignment remains incomplete'
    report['production_code_changed']=False
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    with torch.no_grad():main()
