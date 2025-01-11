import math
import torch.distributed as dist
import torch
import torch.nn as nn
import os
import numpy as np
import inspect
from torch.nn.parallel import DistributedDataParallel as DDP
from dataclasses import dataclass
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd %config.n_head==0
        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head=config.n_head
        self.n_embd=config.n_embd
        self.register_buffer('bias',torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))

    def forward(self,x):
        B,T,C=x.size()
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embd,dim=2)
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        #传统的Attention
        # att=(q@k.transpose(-2,-1)*1.0/math.sqrt(k.size(-1)))
        # att=att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        # att=F.softmax(att,dim=-1)
        # y=att@v
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)#Flash Attention

        y=y.transpose(1,2).contiguous().view(B,T,C)
        y=self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu=nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1
    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)
    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size:int=1024
    vocab_size:int=50257
    n_layer:int=12
    n_head:int=12
    n_embd:int=768


# GPT model
class gpt(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config

        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            wpe=nn.Embedding(config.block_size,config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)

        self.transformer.wte.weight=self.lm_head.weight

        self.apply(self._init_weights)
    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            std=0.02
            if hasattr(module,'NANOGPT_SCALE_INIT'):
                std *=(2*self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,idx,targets=None):
        B,T=idx.size()
        assert T<=self.config.block_size
        pos=torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_emb=self.transformer.wpe(pos)
        tok_emb=self.transformer.wte(idx)
        x=tok_emb+pos_emb
        for block in self.transformer.h:
            x=block(x)
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x)
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))

        return logits,loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        # config_args['bias'] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        config = GPTConfig(**config_args)
        model = gpt(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model

        model_hf = GPT2LMHeadModel.from_pretrained(local_files_only=True,pretrained_model_name_or_path='./')
        sd_hf = model_hf.state_dict()


        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optim(self,weight_decay,lr,device_type):
        #选出需要计算梯度的参数
        param_dict={pn: p for pn,p in self.named_parameters()}
        param_dict={pn: p for pn,p in param_dict.items() if p.requires_grad}
        decay_params=[p for n,p in param_dict.items() if p.dim()>=2]
        no_decay_params=[p for n,p in param_dict.items() if p.dim()<2]
        optim_groups=[
            {'params':decay_params,'weight_decay':weight_decay},
            {'params':no_decay_params,'weight_decay':0.0}
        ]
        num_decay_params=sum(p.numel() for p in decay_params)
        num_no_decay_params=sum(p.numel() for p in no_decay_params)
        if master_process:
            print(f'need decayed param tensors: {len(decay_params)}, with {num_decay_params}')
            print(f'no need decayed param tensors: {len(no_decay_params)}, with {num_no_decay_params}')
        fused_available='fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused=fused_available and device_type=="cuda"
        optimizer=torch.optim.AdamW(optim_groups,lr=lr,betas=(0.9,0.95),eps=1e-8,fused=True)
        return optimizer

# num_return_seq=5
# max_length=30
# #model=gpt.from_pretrained('gpt2')
# model=gpt(GPTConfig)
# model.eval()
# model.to('cuda')


import tiktoken
enc=tiktoken.get_encoding('gpt2')
#测试预训练模型的文本生成效果
# tokens=enc.encode("hello,I'm a language model,")
# tokens=torch.tensor(tokens,dtype=torch.long)
# tokens=tokens.unsqueeze(0).repeat(num_return_seq,1)
# x=tokens.to('cuda')
#
# torch.manual_seed(42)
# torch.cuda.manual_seed(2)
# while x.size(1)<max_length:
#     with torch.no_grad():
#         logits=model(x)
#         logits=logits[:,-1,:]
#         probs=F.softmax(logits,dim=-1)
#         topk_probs,topk_indices=torch.topk(probs,50,dim=-1)
#         ix=torch.multinomial(topk_probs,1)
#         xcol=torch.gather(topk_indices,-1,ix)
#         x=torch.cat((x,xcol),dim=1)
#
# for i in range(num_return_seq):
#     tokens=x[i,:max_length].tolist()
#     decoded=enc.decode(tokens)
#     print('>',decoded)

def load_tokens(name):
    npt=np.load(name)
    ptt=torch.tensor(npt,dtype=torch.long)
    return ptt

class DataLoader1:
    def __init__(self,B,T,process_rank,num_processes,split):
        self.B=B
        self.T=T
        self.process_rank=process_rank
        self.num_processes=num_processes

        with open('input.txt','r') as f:
            text=f.read()

        enc=tiktoken.get_encoding('gpt2')
        tokens=enc.encode(text)
        self.tokens=torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")


        self.current_pos=self.B*self.T*self.process_rank


    def next_batch(self):
        B,T=self.B,self.T
        buf=self.tokens[self.current_pos:self.current_pos+B*T+1]
        x=(buf[:-1]).view(B,T)
        y=(buf[1:]).view(B,T)
        self.current_pos+=B*T*self.num_processes
        if self.current_pos+B*T*self.num_processes+1 > len(self.tokens):
            self.current_pos=self.B*self.T*self.process_rank

        return x,y

from torch.distributed import init_process_group,destroy_process_group


ddp=int(os.environ.get('RANK',-1))!=-1
#多GPU并行训练
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank=int(os.environ["RANK"])
    ddp_local_rank=int(os.environ["LOCAL_RANK"])
    ddp_world_size=int(os.environ["WORLD_SIZE"])
    device=f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process=ddp_rank==0
else:
    ddp_rank=0
    ddp_local_rank=0
    ddp_world_size=1
    master_process=True
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'

device_type="cuda" if device.startswith("cuda") else "cpu"
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

import time

total_batch_size=524288

B,T=8,2048 #设置batch_size和seq_len
assert total_batch_size%(B*T)==0

#grad_accum_steps=total_batch_size // (B*T)
grad_accum_steps=total_batch_size // (B*T*ddp_world_size)
if master_process:
    print(f'batch size: {total_batch_size}, grad accum steps: {grad_accum_steps}')


train_loader=DataLoader1(B,T,process_rank=ddp_rank,num_processes=ddp_world_size,split='train')
val_loader=DataLoader1(B,T,process_rank=ddp_rank,num_processes=ddp_world_size,split='val')

torch.set_float32_matmul_precision('high')

model=gpt(GPTConfig(vocab_size=50304))
model.to(device)
#model=torch.compile(model)
if ddp:
    model=DDP(model,device_ids=[ddp_local_rank])
raw_model=model.module if ddp else model

max_lr=6e-4
min_lr=max_lr*0.1
warmup_steps=10
max_steps=50
def get_lr(it):
    if it<warmup_steps:
        return max_lr *(it+1)/warmup_steps
    if it>max_steps:
        return min_lr
    decay_ratio=(it-warmup_steps)/(max_steps-warmup_steps)
    assert 0<=decay_ratio<=1
    coeff=0.5 *(1.0+math.cos(math.pi*decay_ratio))
    return min_lr+coeff*(max_lr-min_lr)

lr=6e-4
optimizer=raw_model.configure_optim(weight_decay=0.1,lr=lr,device_type=device_type)
#optimizer=torch.optim.AdamW(model.parameters(),lr=lr,betas=(0.9,0.95),eps=1e-8,fused=True)
# scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0=10,T_mult=50,eta_min=lr*0.1)


for step in range(max_steps):
    model.train()
    t0=time.time()
    loss_acc=0.0
    optimizer.zero_grad()
    for m in range(grad_accum_steps):
        x,y=train_loader.next_batch()
        x,y=x.to(device),y.to(device)
        with torch.autocast(device_type=device_type,dtype=torch.bfloat16):#混合精度训练，使用bf16
            logits,loss=model(x,y)
        # logits,loss=model(x,y)#单精度训练
        loss=loss/grad_accum_steps
        loss_acc+=loss.detach()
        if ddp:
            model.require_backward_grad_sync=(m==grad_accum_steps-1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_acc,op=dist.ReduceOp.AVG)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    lr=get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
    optimizer.step()

    torch.cuda.synchronize()
    t1=time.time()
    t_train=(t1-t0)
    token_per_sec=(train_loader.B *train_loader.T *grad_accum_steps*ddp_world_size)/(t1-t0)
    if master_process:
        print(f'step: {step}, loss: {loss_acc.item()},norm: {norm:.5f} ,time: {t_train:.4f}s, token/s: {token_per_sec:.2f}')


# 单卡运行脚本： python train_gpt2.py
# 多卡使用 torchrun 运行脚本: torchrun --standalone --nproc_per_node={num_gpus} train_gpt2.py 或 python -m torch.distributed.launch --nproc_per_node={num_gpus} --master_port={port} train_gpt2.py
