import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.activations import ACT2FN


""" Model Config """


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
        self,
        vocab_size: int = 6400,
        num_hidden_layers: int = 8,
        hidden_dim: int = 768,
        max_seq_len: int = 4096,
        eps: float = 1e-6,
        num_heads: int = 16,
        num_kv_heads: int = 8,
        flash_attn: bool = True,
        dropout: float = 0.0,
        intermediate_dim: int = None,
        hidden_act: str = 'silu',
        use_moe: bool = False,
        aux_loss_alpha: float = 5e-4,
        num_router_experts: int = 4,
        num_shared_experts: int = 1,
        moe_top_k: int = 2
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        # RMSNorm
        self.eps = eps
        # Attention
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_kv_heads = num_kv_heads
        self.flash_attn = flash_attn
        self.dropout = dropout
        # FFN
        self.intermediate_dim = intermediate_dim
        self.hidden_act = hidden_act
        # MoE
        self.use_moe = use_moe
        self.aux_loss_alpha = aux_loss_alpha                # 负载均衡损失的权重(占总损失的占比)
        self.num_router_experts = num_router_experts        # 路由专家的数量
        self.num_shared_experts = num_shared_experts        # 共享专家的数量
        self.moe_top_k = moe_top_k


""" Model Structure """


def rotate_half(x):
    # x.shape = (b, s, n or n_kv, h)
    rotate_x = torch.zeros_like(x)
    rotate_x[..., 0::2] = -x[..., 1::2]
    rotate_x[..., 1::2] = x[..., 0::2]
    return rotate_x


def apply_rotary_pos_emb(q, k, cos, sin):
    # sin, cos.shape = (s, h) --> (1, s, 1, h)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    q_embd = (q * cos) + (rotate_half(q) * sin)
    k_embd = (k * cos) + (rotate_half(k) * sin)
    return q_embd, k_embd


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    # x.shape = (b, s, n_kv, h)
    b, s, n_kv, h = x.size()
    x = x.unsqueeze(3).expand(b, s, n_kv, n_rep, h).reshape(b, s, n_kv * n_rep, h)
    return x


def precompute_freqs_cis(max_pos, dim):
    # 初始化频率矩阵
    F = torch.zeros(max_pos, dim)                                        # F.shape = (max_pos, dim)

    pos = torch.arange(0, max_pos, dtype=torch.float).unsqueeze(1)       # pos.shape = (max_pos, 1)
    i_2 = torch.arange(0, dim, 2, dtype=torch.float).unsqueeze(0)        # i_2.shape = (1, dim // 2)
    inv_freq = 1.0 / 10000 ** (i_2 / dim)                                # inv_freq.shape = (1, dim // 2)
    freqs = pos * inv_freq                                               # pos.shape = (max_pos, dim // 2)

    # 更新频率矩阵的值
    F[:, 0::2] = freqs
    F[:, 1::2] = freqs
    return F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x.shape = (b, s, h)
        hidden_states = x.float()
        hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        hidden_states = hidden_states * self.weights
        return hidden_states.type_as(x)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_kv_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.dropout = config.dropout

        self.query = nn.Linear(config.hidden_dim, config.num_heads * config.head_dim)
        self.key = nn.Linear(config.hidden_dim, config.num_kv_heads * config.head_dim)
        self.value = nn.Linear(config.hidden_dim, config.num_kv_heads * config.head_dim)
        self.flash_attn = config.flash_attn
        self.attn_dropout = nn.Dropout(config.dropout)

        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_dim)
        self.o_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_kv_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # x.shape = (b, s, d)
        b, s, _ = x.size()
        # Calculate Q, K, V
        q, k, v = (
            self.query(x).view(b, s, self.num_heads, self.head_dim),      # q.shape = (b, s, n, h)
            self.key(x).view(b, s, self.num_kv_heads, self.head_dim),     # k.shape = (b, s, n_kv, h)
            self.value(x).view(b, s, self.num_kv_heads, self.head_dim),   # v.shape = (b, s, n_kv, h)
        )

        # Rotate Q, K
        # sin, cos.shape = (s, h)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Update KV Cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        present_key_value = (k, v) if use_kv_cache else None

        q, k, v = (
            q.transpose(1, 2),                      # q.shape = (b, n, s, h)
            repeat_kv(k, self.n_rep).transpose(1, 2),           # k.shape = (b, n, s, h)
            repeat_kv(v, self.n_rep).transpose(1, 2)            # v.shape = (b, n, s, h)
        )

        # Attention
        # 训练阶段: s > 1 且不使用 kv_cache
        # 推理阶段: s = 1 且使用 kv_cache
        # torch.all(attention_mask == 1): 判断 attention_mask 中所有元素是否都为 1
        if self.flash_attn and s > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # scores.shape = (b, n, s, s)
            scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
            # 当 s = 1 的时候就没有 mask 了因为 diagonal=1
            scores = scores + torch.triu(
                torch.full((s, s), float("-inf"), device=scores.device),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(1)

            if attention_mask is not None:
                # attention_mask.shape = (b, s)
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # output.shape = (b, n, s, h)
            scores = F.softmax(scores, dim=-1)
            scores = self.attn_dropout(scores)
            output = scores @ v

        # output.shape = (b, s, n, h)
        output = output.transpose(1, 2).reshape(b, s, -1)
        output = self.o_dropout(self.o_proj(output))
        return output, present_key_value


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.intermediate_dim is None:
            intermediate_dim = int(config.hidden_dim * 8 / 3)
            config.intermediate_dim = 64 * ((intermediate_dim + 64 - 1) // 64)

        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # FFN: down(σ(gate(x)) * up(x))
        x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.dropout(x)


class MOERouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alpha = config.aux_loss_alpha
        self.top_k = config.moe_top_k
        self.num_router_experts = config.num_router_experts

        self.router = nn.Linear(config.hidden_dim, self.num_router_experts)

    def forward(self, x):
        # x.shape = (b, s, d)
        b, s, d = x.size()
        # x.shape = (b * s, d)
        x = x.view(b * s, -1)

        # router_logits.shape = (b * s, n_r)
        router_logits = self.router(x)
        router_logits = F.softmax(router_logits, dim=-1)

        # router_weights, select_expert_ids.shape = (b * s, top_k)
        router_weights, select_expert_ids = torch.topk(router_logits, k=self.top_k, dim=-1)
        router_weights = router_weights / (router_weights.sum(dim=-1, keepdim=True) + 1e-20)

        if self.training and self.alpha > 0:
            # expert_mask.shape = (b * s * top_k, n_r)
            expert_mask = F.one_hot(select_expert_ids.view(-1), num_classes=self.num_router_experts)
            # ci, fi.shape = (n_r)
            ci = expert_mask.float().mean(dim=0)
            fi = ci * self.num_router_experts

            # pi.shape = (n_r)
            pi = router_logits.mean(dim=0)

            # aux_loss
            aux_loss = (fi * pi).sum() * self.alpha
        else:
            aux_loss = 0

        return router_weights, select_expert_ids, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.moe_top_k
        self.num_router_experts = config.num_router_experts
        self.num_shared_experts = config.num_shared_experts

        self.router = MOERouter(config)
        self.router_experts = nn.ModuleList([
            FeedForward(config) for _ in range(self.num_router_experts)
        ])
        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config) for _ in range(self.num_shared_experts)
            ])

    def forward(self, x):
        # x.shape = (b, s, d)
        b, s, d = x.size()
        # router_weights, select_expert_ids.shape = (b * s, top_k)
        router_weights, select_expert_ids, aux_loss = self.router(x)

        # ========== RouterExpert ==========
        hidden_states = x.view(-1, d)
        hidden_states = hidden_states.repeat_interleave(self.top_k, dim=0)  # (b * s * top_k, d)
        y = torch.zeros_like(hidden_states, dtype=hidden_states.dtype)
        select_expert_ids = select_expert_ids.view(-1)                      # (b * s * top_k)
        
        for i, expert in enumerate(self.router_experts):
            y[select_expert_ids == i, :] = expert(hidden_states[select_expert_ids == i, :]).to(y.dtype)

        y = y.view(b * s, self.top_k, -1)                           # (b * s, top_k, d)
        router_weights = router_weights.unsqueeze(dim=-1)           # (b * s, top_k, 1)
        y = (y * router_weights).sum(dim=1)                         # (b * s, d)
        y = y.view(b, s, -1)                                        # (b, s, d)

        # ========== SharedExpert ==========
        if self.num_shared_experts > 0:
            y_s = torch.cat([
                expert(x).unsqueeze(-2) for expert in self.shared_experts         # (b, s, 1, d)
            ], dim=-2)                                                            # (b, s, n_s, d)
            y_s = y_s.sum(dim=-2)                                                 # (b, s, d)

            y = y + y_s

        # ========== Output ==========
        self.aux_loss = aux_loss
        return y


class MiniMindBlock(nn.Module):
    def __init__(self, layer_idx, config):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = Attention(config)
        self.attn_norm = RMSNorm(config.hidden_dim, config.eps)
        self.ffn_norm = RMSNorm(config.hidden_dim, config.eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_kv_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        residual = x
        x, present_key_value = self.attn(
            self.attn_norm(x),
            position_embeddings,
            past_key_value,
            use_kv_cache,
            attention_mask
        )
        x = x + residual
        x = x + self.mlp(self.ffn_norm(x))
        return x, present_key_value


class MiniMindModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([
            MiniMindBlock(idx, config) for idx in range(config.num_hidden_layers)
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = RMSNorm(config.hidden_dim, config.eps)

        # Rotary Position Embedding
        freqs = precompute_freqs_cis(config.max_seq_len, config.head_dim)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_kv_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs            # 兼容 transformer.generator 的其他参数
    ):
        # x.shape = (b, s)
        _, s = input_ids.size()
        # hidden_states.shape = (b, s, d)
        hidden_states = self.token_embeddings(input_ids)

        # 处理 past_key_values
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values if past_key_values is not None else [None] * len(self.layers)

        # 处理位置编码
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        position_embeddings = (
            self.cos_cached[start_pos: start_pos + s, :],
            self.sin_cached[start_pos: start_pos + s, :]
        )

        # 正片
        present_key_values = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present_key_value = layer(
                hidden_states, position_embeddings, past_key_value, use_kv_cache, attention_mask
            )
            present_key_values.append(present_key_value)

        hidden_states = self.norm(hidden_states)
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )
        return hidden_states, present_key_values, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_dim, self.config.vocab_size, bias=False)
        # 词嵌入矩阵和输出头共享权重
        self.model.token_embeddings.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,   # 默认保留全部
        labels=None,
        **kwargs                                        # 兼容 transformer.generator 的其他参数
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_kv_cache=use_cache,
            **kwargs
        )

        # slice 切片:
        # slice(start, end)                 <==>    [start: end]
        # slice(start, None)                <==>    [start:]
        # slice(-logits_to_keep, None)      <==>    [-logits_to_keep:]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None

        if labels is not None:
            x, y = logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                x.view(-1, x.size(-1)),
                y.view(-1),
                ignore_index=-100
            )

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

    @torch.inference_mode()
    def generate(
        self,
        inputs=None,
        attention_mask=None,
        max_new_tokens=8192,
        temperature=0.85,                   # 调整概率分布的尖锐度: 当 temperature < 1 会放大每个位置的差距; temperature > 1 会缩小每个位置的差距
        repetition_penalty=1.0,             # 对当前序列中已经出现过的 token 做降权, 减少它们被再次采样的概率
        top_k=50,                           # 筛选: 每一步生成时, 只保留权重最大的 top_k 个采样位置, 其他位置的权重都变为 -float('inf')
        top_p=0.85,                         # 筛选: 只在累计概率达到 top_p 的这些位置进行采样, 其他位置的权重都变为 -float('inf')
        eos_token_id=2,
        streamer=None,
        use_cache=True,
        num_return_sequences=1,             # 生成 num_return_sequences 份回答
        do_sample=True,
        **kwargs
    ):
        input_ids = kwargs.pop('input_ids', inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop('past_key_values', None)
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())

        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values is not None else 0
            outputs = self.forward(
                input_ids=input_ids[:, past_len:],
                past_key_values=past_key_values,
                use_cache=use_cache,
                attention_mask=attention_mask,
                **kwargs
            )
            # 给新生成的位置的 attention_mask 标记为 1
            attention_mask = torch.cat([
                attention_mask,
                attention_mask.new_ones(attention_mask.size(0), 1)
            ], dim=-1) if attention_mask is not None else None
            # logits.shape = (b, vocab_size)
            logits = outputs.logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                for i in range(logits.size(0)):
                    logits[i, torch.unique(input_ids[i, :])] /= repetition_penalty

            if top_k > 0:
                # 每行的生成权重总大到小排序后, 第 top_k 个大的权重。将小于该权重的位置的权重置为 -float('inf')
                # unsqueeze(-1) 保证维度对应, 为 (b, 1)
                row_min = torch.topk(logits, top_k)[0][:, -1].unsqueeze(-1)
                logits[logits < row_min] = -float('inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
                mask = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1),
                    dim=-1
                ) > top_p
                # 整体向右移动一位, 因为要保留刚好累积概率达到 top_p 的位置
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

            # num_samples: 要选择几个样本, 自回归模型就每次生成一个新 token 因此 num_samples = 1
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),                                                 # condition
                    next_token.new_full((next_token.size(0), 1), eos_token_id),        # if condition: 让已经结束生成的那条答案, 不管新生成的 token 是什么都强制为 eos_token
                    next_token                                                              # if not condition: 还没生成完的那些答案继续生成
                )
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break

        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids


if __name__ == '__main__':
    """ 代码实测 """
    # batch_size, seq_len = 16, 256
    # config = MiniMindConfig()
    # input_ids = torch.randint(0, config.vocab_size, size=(batch_size, seq_len))
    # hidden_states = torch.randn((batch_size, seq_len, config.hidden_dim), dtype=torch.float)
    # cos, sin = (
    #     torch.randn((seq_len, config.head_dim), dtype=torch.float),
    #     torch.randn((seq_len, config.head_dim), dtype=torch.float)
    # )

    # print(f"Input shape: {input_ids.shape}")
    # print(f"Hidden states shape: {hidden_states.shape}")

    # attn = Attention(config)
    # ffn = FeedForward(config)
    # block = MiniMindBlock(0, config)
    # moe_router = MOERouter(config)
    # moe_ffn = MOEFeedForward(config)
    # model = MiniMindModel(config)
    
    # hidden_states, _ = attn(x=hidden_states, position_embeddings=(cos, sin))
    # hidden_states, _ = block(x=hidden_states, position_embeddings=(cos, sin))
    # router_weights, select_expert_ids, aux_loss = moe_router(hidden_states)
    # hidden_states = moe_ffn(hidden_states)
    # print(f"hidden_states.shape: {hidden_states.shape}")
    # print(f"aux_loss: {moe_ffn.aux_loss}")
    # hidden_states, _ = model(input_ids)

    # x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # print(x)
    #
    # y = x.new_full((x.size(0), 1), 2)
    # print(y)


