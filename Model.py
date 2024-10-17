'''
1 TinyStory transformer
2 masked linear
'''

import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import cvxpy as cp

np.random.seed(1634)

@dataclass
class FixedLengthModelArgs:
    T: int = 6
    v_ctx: int=152
    v_nt: int=152
    d_input: int=64
    d_hidden: int=128

@dataclass
class TFMModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    #vocab_size: int = 32000
    v_ctx: int = None
    v_nt: int = None
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    pos_enc: str = "rope"
    if_ar: bool = False
    if_extra_ln: int = None


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: TFMModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = (args.dim // args.n_heads //2)*2
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.pos_enc = args.pos_enc

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        if self.pos_enc == "rope":
            # print(self.pos_enc)
            xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            # hidden_dim = int(2 * hidden_dim / 3)
            # hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: TFMModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)



    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: TFMModelArgs):
        super().__init__()
        self.params = params
        #self.vocab_size = params.vocab_size
        self.v_nt = params.v_nt
        self.v_ctx = params.v_ctx
        self.n_layers = params.n_layers
        self.if_ar = params.if_ar

        self.tok_embeddings = nn.Embedding(params.v_ctx, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        self.if_extra_ln = params.if_extra_ln
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        if params.if_extra_ln is not None:
            self.pre_output = nn.Linear(params.dim, params.if_extra_ln, bias=False)
            self.norm = RMSNorm(params.if_extra_ln, eps=params.norm_eps)
            self.output = nn.Linear(params.if_extra_ln, params.v_nt, bias=False)
        else:
            self.norm = RMSNorm(params.dim, eps=params.norm_eps)
            self.output = nn.Linear(params.dim, params.v_nt, bias=False)

        # share the unembedding parameters with the embedding parameters
        #self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None
        self.positionwise_loss = None

        self.name = f"{self.n_layers} layer Transformer pos {self.params.pos_enc}"

        self.last_emb = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        if self.if_extra_ln is not None:
            h = self.pre_output(h)
        h = self.norm(h)

        # self.last_emb = h

        # if we are given some desired targets also calculate the loss
        if self.if_ar:
            logits = self.output(h)
            # get loss at each position
            non_reduction_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction="none")
            # dimension of non_reduction_loss is (bsz*seqlen)
            self.positionwise_loss = non_reduction_loss.view(targets.size()).sum(dim=0)
            # dimension of positionwise_loss is (seqlen)
            self.last_loss = self.positionwise_loss.sum()
        else:
            embeds = torch.squeeze(h[:, [-1], :])
            logits = self.output(embeds)
            #logits_last = logits
            #self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            self.last_loss = F.cross_entropy(logits, targets, reduction='sum')


        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def get_embeddings(self, tokens: torch.Tensor, v_ctx2v, id_ctx_dict):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        if self.if_extra_ln is not None:
            h = self.pre_output(h)
        h = self.norm(h)
        embed = torch.squeeze(h[:, [-1], :])
        emb_dict = dict()

        for k, id_ctx in id_ctx_dict.items():
            emb_dict[k] = embed[id_ctx["id"]].cpu().detach().numpy()

        # for i in range(embed.shape[0]):
        #     tokens_in_v = v_ctx2v[tokens[i]]
        #     token_byte = tokens_in_v.astype(np.uint16).tobytes()
        #     emb_dict[token_byte] = embed[i].cpu().detach().numpy()

        return emb_dict

    def forward_embedding(self, tokens, t=-1):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        embed = torch.squeeze(h[:, [t], :])

        return embed

# basic mlp module
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class FixedLengthMLP(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, T, v_ctx, v_nt, d_input, d_hiddens, norm_eps=1e-5):
        super().__init__()
        self.name = "FixedLMLP"
        self.d_input, self.d_hiddens = d_input, d_hiddens

        self.layers = nn.ModuleList()

        self.tok_embeddings = nn.Embedding(v_ctx, d_input)
        #self.mlp = MLP(input_size=d_input, hidden_size=d_hidden, output_size=d_output)
        # self.output = nn.Linear(d_output, v_nt, bias=False)
        self.apply(self._init_weights)

        current_size = self.d_input * (T-1)

        for size in self.d_hiddens:
            self.layers.append(nn.Linear(current_size, size))
            self.layers.append(RMSNorm(size, eps=norm_eps))
            current_size = size

        self.output = nn.Linear(current_size, v_nt)
        self.norm = RMSNorm(current_size, eps=norm_eps)

        self.last_emb = None


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, y):

        h = self.tok_embeddings(tokens)
        # Forward pass through each layer
        h = h.reshape(h.shape[0], -1)
        for layer in self.layers:
            h = F.relu(layer(h))  # Using ReLU activation for hidden layers
            # h = self.norm(h)

        # embeds = self.norm(h)
        embeds = h
        # Output layer - no activation; logits are returned
        logits = self.output(embeds)

        self.last_loss = F.cross_entropy(logits, y, reduction='sum')

        return logits

    # def get_embeddings(self, tokens: torch.Tensor, v_ctx2v, id_ctx_dict):
    #     h = self.tok_embeddings(tokens)
    #     h = h.reshape(h.shape[0], -1)
    #     # Forward pass through each layer
    #     for layer in self.layers:
    #         h = self.norm(h)
    
    #         # embeds = self.norm(h)
    #     embeds = h
    #     # Output layer - no activation; logits are returned
    #     logits = self.output(embeds)
    
    #     return logits

    def forward_embedding(self, tokens):
        h = self.tok_embeddings(tokens)
        h = h.reshape(h.shape[0], -1)
        # Forward pass through each layer
        for layer in self.layers:
            h = F.relu(layer(h))  # Using ReLU activation for hidden layers
            # h = self.norm(h)

        # embeds = self.norm(h)
        embeds = h

        return embeds

class MultiLabelSVM(nn.Module):
    def __init__(self, emb_dict, support_dict, prob_dict, d, v_nt):
        super().__init__()
        self.emb_dict = emb_dict
        self.support_dict = support_dict
        self.prob_dict = prob_dict

        self.d = d
        self.v_nt = v_nt


    def find_Wfin(self):
        W_fin_cvx = cp.Variable((self.v_nt, self.d))
        obj_9 = cp.Minimize(cp.norm2(W_fin_cvx))
        constraint_list = []
        for k, h in self.emb_dict.items():
            for ind_z, z in enumerate(self.support_dict[k]):
                # ind_z = np.where(self.support_dict[k] == z)
                for ind_z_, z_ in enumerate(self.support_dict[k]):
                    if z_ == z:
                        continue
                        # ind_z_ = np.where(self.support_dict[k]==z_)[0][0]
                    constraint_list.append(
                        (W_fin_cvx[z] - W_fin_cvx[z_]).T @ (h) \
                        == cp.log(self.prob_dict[k][ind_z]) - cp.log(self.prob_dict[k][ind_z_]))

        const_9 = constraint_list
        prob_9 = cp.Problem(obj_9, const_9)

        result = prob_9.solve(verbose=True)
        W_fin = W_fin_cvx.value
        # W_star = np.random.normal(size=(v,d))

        if np.isnan(result):
            print('no solution found for W_fin')

        # check_list = []
        # for j in range(h_s.shape[0]):
        #     for z in s_s[j]:
        #         for z_ in range(v):
        #             if z_ == z:
        #                 continue
        #             elif z_ in s_s[j]:
        #                 check_list.append(abs((W_star[z] - W_star[z_]).T@(h_s[j])) <= 1e-10)
        #             else:
        #                 check_list.append((W_star[z] - W_star[z_]).T@(h_s[j]) > 1)

        return W_fin

    def find_WStar(self):
        W_star_cvx = cp.Variable((self.v_nt, self.d))
        obj_10 = cp.Minimize(cp.norm2(W_star_cvx))
        constraint_list = []
        for k, h in self.emb_dict.items():
            for ind_z, z in enumerate(self.support_dict[k]):
                for z_ in range(self.v_nt):
                    if z_ == z:
                        continue
                    elif z_ in self.support_dict[k]:
                        constraint_list.append((W_star_cvx[z] - W_star_cvx[z_]).T @ (h) == 0)
                    else:
                        constraint_list.append((W_star_cvx[z] - W_star_cvx[z_]).T @ (h) >= 1)

        const_10 = constraint_list
        prob_10 = cp.Problem(obj_10, const_10)

        result = prob_10.solve(cp.SCS, verbose=True)
        W_star = W_star_cvx.value
        # W_star = np.random.normal(size=(v,d))

        if np.isnan(result):
            print('no solution found for W_star')

        # check_list = []
        # for j in range(h_s.shape[0]):
        #     for z in s_s[j]:
        #         for z_ in range(v):
        #             if z_ == z:
        #                 continue
        #             elif z_ in s_s[j]:
        #                 chewck_list.append(abs((W_star[z] - W_star[z_]).T@(h_s[j])) <= 1e-10)
        #             else:
        #                 check_list.append((W_star[z] - W_star[z_]).T@(h_s[j]) > 1)

        return W_star

# define network
class UFM(nn.Module):
    last_loss: Optional[torch.Tensor]
    def __init__(self, m, d, k):
        super(UFM, self).__init__()
        self.name = "UFM"
        self.m = m
        self.l1 = nn.Linear(m, d, bias=False)
        self.output = nn.Linear(d, k, bias=False)
        # self.double()
        self.last_emb = None

    def forward(self, x, y, return_embed = False):
        embeds = self.l1(x)
        # self.last_emb = embeds
        logits = self.output(embeds)

        self.last_loss = F.cross_entropy(logits, y, reduction='sum')

        if return_embed:
            return embeds, logits

        return logits

    def forward_embedding(self, x):
        embeds = self.l1(x)

        return embeds

    def get_embeddings(self, id_ctx_dict):
        tokens = torch.from_numpy(np.eye(self.m)).float()
        embeds = self.l1(tokens)

        emb_dict = {}
        for k, id_ctx in id_ctx_dict.items():
            emb_dict[k] = embeds[id_ctx["id"]].cpu().detach().numpy()
        return emb_dict



#define a LSTM network
class LSTMModel(nn.Module):
    def __init__(self, T, v_ctx, v_nt, d_input, d_hiddens, num_layers=2, norm_eps=1e-5):
        super().__init__()
        self.name = "LSTM"
        self.d_input = d_input
        self.d_hiddens = d_hiddens
        self.num_layers = num_layers

        # Embedding layer
        self.tok_embeddings = nn.Embedding(v_ctx, d_input)
        # LSTM
        self.lstm = nn.LSTM(input_size=d_input, hidden_size=d_hiddens[0], num_layers=num_layers, batch_first=True)
        # Output layer: logits
        self.output = nn.Linear(d_hiddens[0], v_nt)
        # Normalization 
        self.norm = RMSNorm(d_hiddens[0], eps=norm_eps)

        # Initialize the weights
        self.apply(self._init_weights)
        self.last_emb = None  # Stores the last embeddings

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens, y=None):

        h = self.tok_embeddings(tokens)
        # Initialize hidden and cell states with zeros
        h_0 = torch.zeros(self.num_layers, h.size(0), self.d_hiddens[0]).to(h.device)
        c_0 = torch.zeros(self.num_layers, h.size(0), self.d_hiddens[0]).to(h.device)

        out, _ = self.lstm(h, (h_0, c_0))

        # Get the last hidden state for each sequence in the batch
        last_hidden_state = out[:, -1, :]
        embeds = self.norm(last_hidden_state)
        self.last_emb = embeds

        logits = self.output(embeds)

        if y is not None:
            self.last_loss = F.cross_entropy(logits, y, reduction='sum')

        return logits

    def forward_embedding(self, tokens):

        h = self.tok_embeddings(tokens)

        h_0 = torch.zeros(self.num_layers, h.size(0), self.d_hiddens[0]).to(h.device)
        c_0 = torch.zeros(self.num_layers, h.size(0), self.d_hiddens[0]).to(h.device)
        out, _ = self.lstm(h, (h_0, c_0))

        last_hidden_state = out[:, -1, :]
        embeds = self.norm(last_hidden_state)
        return embeds
    
    def get_embeddings(self, tokens: torch.Tensor, v_ctx2v, id_ctx_dict: dict) -> dict:
        embeddings = self.forward_embedding(tokens)

        emb_dict = {}

        for key, id_ctx in id_ctx_dict.items():
            emb_dict[key] = embeddings[id_ctx["id"]].cpu().detach().numpy()

        return emb_dict