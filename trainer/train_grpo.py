import os
import sys
import warnings

warnings.filterwarnings('ignore')

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel, AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler, LMForRewardModel


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()
                rewards[response_idx] -= rep_penalty(answer)

                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def gen_per_token_logps(model, input_ids, attention_mask, n_keep):
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=n_keep + 1,
    ).logits

    logits = logits[:, :-1, :]          # (B * num_gens, R, vocab_size)
    labels = input_ids[:, -n_keep:]     # (B * num_gens, R)

    # per_token_logps = []
    # for (
    #     lgt,    # (R, vocab_size)
    #     lb      # (R, )
    # ) in zip(logits, labels):
    #     log_p = (
    #         F.log_softmax(lgt, dim=-1)
    #         .gather(dim=-1, index=lb.unsqueeze(-1))
    #         .squeeze(-1)
    #     )
    #     per_token_logps.append(log_p)
    #
    # return torch.stack(per_token_logps, dim=0)

    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    return per_token_logps


def grpo_train_epoch(epoch, loader, iters, ref_model, reward_model, start_step=0, wandb=None):
    for step, batch in enumerate(loader, start=start_step + 1):
        # 编码输入
        prompts = batch['prompt']
        enc = tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
            add_special_tokens=False,
            return_token_type_ids=False,
            padding_side="left"  # 这里使用左填充, 这样所有样本的真实句子都在最右侧, generate() 总是会从最后一个真实 token 而非 padding token 后面生成句子
        ).to(args.device)

        # ========== 使用 Policy Model 生成样本 ==========
        with torch.no_grad():
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            # gen_out.shape = (B * num_gens, P + R)
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=args.num_generations,
            )

        gen_out = gen_out.detach().clone() if gen_out.is_inference() else gen_out
        prompt_length = enc.input_ids.shape[1]
        # completion_ids.shape = (B * num_gens, R)
        completion_ids = gen_out[:, prompt_length:]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # ========== 计算采样 token 在 Policy Model 中的 logp ==========
        # valid_res_mask.shape = (B * num_gens, R)
        resp_non_pad = (completion_ids != tokenizer.pad_token_id).long()
        eos_seen = (completion_ids == tokenizer.eos_token_id).cumsum(dim=-1)
        valid_res_mask = resp_non_pad * (eos_seen <= 1).long()
        # full_mask.shape = (B * num_gens, P + R)
        full_mask = (gen_out != tokenizer.pad_token_id).long()
        full_mask[:, prompt_length:] = full_mask[:, prompt_length:] * valid_res_mask.long()
        with (autocast_ctx):
            # actor_token_logps.shape = (B * num_gens, R)
            actor_token_logps = gen_per_token_logps(actor_model, gen_out, full_mask, completion_ids.size(-1))
            res = actor_model(
                input_ids=gen_out, attention_mask=full_mask,
            ) if lm_config.use_moe else None
            aux_loss = res.aux_loss if res is not None else torch.tensor(0.0, device=args.device)

        # ========== 计算采样 token 在 Old Policy Model 和 Reference Model 中的 logp ==========
        with torch.no_grad():
            # ref_token_logps.shape = (B * num_gens, R)
            ref_token_logps = gen_per_token_logps(ref_model, gen_out, full_mask, completion_ids.size(-1))
            # old_actor_token_logps.shape = (B * num_gens, R)
            old_actor_token_logps = gen_per_token_logps(old_actor_model, gen_out, full_mask, completion_ids.size(-1))

        # ========== 使用 Reward Model 计算奖励 ==========
        # rewards.shape = (B * num_gens, )
        rewards = calculate_rewards(prompts, completions, reward_model)

        grouped_rewards = rewards.view(-1, args.num_generations)
        rewards_mean = grouped_rewards.mean(dim=-1).repeat_interleave(args.num_generations)
        rewards_std = grouped_rewards.std(dim=-1).repeat_interleave(args.num_generations)
        advantages = (rewards - rewards_mean) / (rewards_std + 1e-8)
        advantages = torch.clamp(advantages, -10, 10)

        # ========== 计算 GRPO 损失 ==========
        ratio = torch.exp(actor_token_logps - old_actor_token_logps)
        surr1 = ratio * advantages.unsqueeze(-1)
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages.unsqueeze(-1)

        kl_div = actor_token_logps - ref_token_logps
        kl = torch.exp(kl_div) - kl_div - 1

        per_token_loss = -(torch.min(surr1, surr2) - args.beta * kl)
        policy_loss = (per_token_loss * valid_res_mask).sum(dim=-1) / valid_res_mask.sum(dim=-1)
        policy_loss = policy_loss.mean()

        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        if step % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.update_old_actor_freq == 0:
            state_dict = (
                actor_model.module.state_dict()
                if isinstance(actor_model, DistributedDataParallel)
                else actor_model.state_dict()
            )
            old_actor_model.load_state_dict(state_dict)

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = valid_res_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Actor Loss: {policy_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, Reward: {avg_reward_val:.4f}, '
                   f'Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": avg_reward_val,
                    "avg_response_len": avg_len_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr
                })

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_dim}{moe_suffix}.pth'
            raw_model = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=optimizer, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            actor_model.train()
            del state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="初始学习率")
    parser.add_argument("--from_weight", type=str, default="full_sft", help="加载哪个模型")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--update_old_actor_freq", type=int, default=1, help="更新old_actor_model的频率")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=4, help="每个prompt生成的样本数")
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO裁剪参数（控制策略更新幅度）")
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument("--reward_model_path", type=str, default="../reward_model/internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_dim=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 5. 加载模型
    # 策略模型(Policy Model)
    actor_model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 旧策略模型(Old Policy Model)
    old_actor_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    old_actor_model.eval().requires_grad_(False)
    # 奖励模型(Reward Model)
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=dtype)
    # 参考模型(Reference Model)
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval().requires_grad_(False)

    # 6. 加载数据集和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"cos_cached", "sin_cached"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers,
                                pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + start_step + 1, ref_model, reward_model, start_step, wandb)
        else:
            loader = DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, drop_last=False, shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler)
            grpo_train_epoch(epoch, loader, len(loader), ref_model, reward_model, 0, wandb)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()