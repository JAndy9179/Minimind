import os
import sys
import requests
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer
from contextlib import nullcontext

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def compute_per_token_logps(model, input_ids: torch.Tensor, n_keep: int, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)

    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    logits = unwrapped(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]

    per_token_logps = []
    labels = input_ids[:, -n_keep:]
    for logits_row, ids_row in zip(
        logits,     # (B * num_gen, R, V)
        labels      # (B * num_gen, R)
    ):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            torch.gather(
                logits_row,
                dim=-1,
                index=ids_row.unsqueeze(-1),
            ).squeeze(-1)
        )

    return torch.stack(per_token_logps)


@dataclass
class RolloutResult:
    """ rollout 结果样式 """
    output_ids: torch.Tensor           # (B * num_gen, P + R)
    completion_ids: torch.Tensor       # (B * num_gen, R)
    per_token_logps: torch.Tensor      # (B * num_gen, R)
    completions: List[str]             # len(completions) = B * num_gen
    prompt_lens: torch.Tensor          # (B * num_gen, )
    completion_masks: torch.Tensor     # (B * num_gen, R)


class RolloutEngine(ABC):
    """ Rollout Engine 抽象基类 """

    @abstractmethod
    def rollout(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        num_generations: int,
        temperature: float = 0.8
    ) -> RolloutResult:
        pass

    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        pass


class TorchRolloutEngine(RolloutEngine):
    def __init__(
        self,
        policy_model: torch.nn.Module,
        tokenizer,
        autocast_ctx=None
    ):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.autocast_ctx = autocast_ctx if autocast_ctx else nullcontext()

    def rollout(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        num_generations: int,
        temperature: float = 0.8
    ) -> RolloutResult:
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        with torch.no_grad(), self.autocast_ctx:
            # output.shape = (B * num_gen, P + R)
            output_ids = model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=num_generations,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            prompt_lens = prompt_ids.size(1)
            # output.shape = (B * num_gen, R)
            completion_ids = output_ids[:, prompt_lens:]

            # 生成 mask, 用于 mask prompt 中的 pad_token_ids 和 response 中作为 pad 的 eos_token_id
            # 在 train_agent.py 场景下, 由于 prompt_ids 是单条样本且并没有填充到最大长度, 且模型回答的时候也不会用 eos_token_id 填充, 因此好像可以不用这么麻烦处理
            # full_mask.shape = (B * num_gen, P + R)
            # completion_mask.shape = (B * num_gen, R)
            full_mask = (output_ids != self.tokenizer.pad_token_id).long()
            resp_mask = (completion_ids == self.tokenizer.eos_token_id).long()
            valid_resp_mask = torch.cumsum(resp_mask, dim=-1) <= 1
            completion_masks = valid_resp_mask * (
                completion_ids != self.tokenizer.pad_token_id
            ).long()
            full_mask[:, prompt_lens:] = full_mask[:, prompt_lens:] * valid_resp_mask.long()

            per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1), attention_mask=full_mask)

        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(
            output_ids=output_ids,
            completion_ids=completion_ids,
            per_token_logps=per_token_logps,
            completions=completions,
            prompt_lens=prompt_ids.new_full((output_ids.size(0), ), prompt_lens),
            completion_masks=completion_masks
        )

    def update_policy(self, model: torch.nn.Module):
        self.policy_model = model


def create_rollout_engine(
    engine_type: str = 'torch',
    policy_model: torch.nn.Module = None,
    tokenizer=None,
    device: str = 'cuda',
    autocast_ctx=None,
    sglang_base_url: str = None,
    sglang_model_path: str = None,
    sglang_shared_path: str = None,
) -> RolloutEngine:
    if engine_type == 'torch':
        return TorchRolloutEngine(policy_model, tokenizer, autocast_ctx)
    elif engine_type == 'sglang':
        raise ValueError(f"不支持的引擎类型: sglang")
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")
