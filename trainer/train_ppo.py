import os
import sys

# 📚 Python模块系统
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel


class CriticModel(MiniMindForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.value_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])

        # values.shape = (b, s)
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            answer = response
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            if '</think>' in response:
                thinking_content, answer_content = response.split('</think>', 1)
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
                answer = answer_content.strip()
            rewards[i] -= rep_penalty(answer)

            score = reward_model.get_score(messages, answer)
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    # rewards.shape = (B * num_gen, )
    return rewards


def ppo_train_epoch(epoch, loader, iters, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step=0, wandb=None):
    actor_model.train()
    critic_model.train()

    for step, batch in enumerate(loader):
        prompts = batch['prompt']
        enc = tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
            padding_side="left"     # 这里使用左填充, 这样所有样本的真实句子都在最右侧, generate() 总是会从最后一个真实 token 而非 padding token 后面生成句子
        ).to(args.device)

        # ========== 使用 Policy Model 生成样本 ==========
        with torch.no_grad():
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            # gen_out.shape = (B * num_gen, P + R)
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len,
                do_sample=True,  # True: 按概率分布随机抽下一个 token; False: 总是选择概率最大的 token
                temperature=0.8,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        gen_out = gen_out.detach().clone() if gen_out.is_inference() else gen_out
        prompt_len = enc.input_ids.size(1)
        completion_ids = gen_out[:, prompt_len:]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # ========== 使用 Reward Model 和规则函数计算奖励 ==========
        # rewards.shape = (B * num_gen, )
        rewards = calculate_rewards(prompts, completions, reward_model)
        # full_mask.shape = (B * num_gen, P + R)
        # 由于回答部分是用 eos_token_id 做填充, 因此需要 mask 掉 prompt 中的 pad_token_id 和回答部分的作为填充的 eos_token_id
        response_mask = (gen_out[:, prompt_len:] == tokenizer.eos_token_id).long()
        valid_response = response_mask.cumsum(dim=-1) <= 1
        full_mask = (gen_out != tokenizer.pad_token_id).long()
        full_mask[:, prompt_len:] = full_mask[:, prompt_len:] * valid_response.long()
        # value_seq.shape = (B * num_gen, P + R)
        value_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)
        # last_indices.shape = (B * num_gen, )
        last_indices = (
            full_mask * torch.arange(value_seq.size(1), device=gen_out.device)
        ).argmax(dim=-1)
        # values = (B * num_gen, )
        # 使用每个样本最后一个非 padding token 对应的 critic 输出该样本的期望奖励, 并根据 Reward 计算样本级 advantage
        values = value_seq[torch.arange(value_seq.shape[0], device=value_seq.device), last_indices]
        # advantages = (B * num_gen, )
        advantages = rewards - values

        # ========== 计算采样的 token 在 actor_model 中的预测概率 ==========
        # labels.shape = (B * num_gen, P + R - 1)
        labels = gen_out[:, 1:].clone()
        with autocast_ctx:
            res = actor_model(
                input_ids=gen_out, attention_mask=full_mask
            )
            actor_logits = res.logits
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)

        # actor_token_logps.shape = (B * num_gen, P + R - 1)
        actor_token_logps = F.log_softmax(actor_logits[:, :-1, :], dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # final_mask.shape = (B * num_gen, P + R - 1)
        # 因为 prompt 的最后一个 token 是用来预测第一个 response token 的, 所以这一位也要保留
        resp_mask = torch.arange(actor_token_logps.size(1), device=gen_out.device).unsqueeze(0) >= (prompt_len - 1)
        final_mask = full_mask[:, 1:].clone() & resp_mask
        # actor_logps.shape = (B * num_gen, )
        # 将 response 中各个 token 的 log 概率加起来, 得到每条序列的总 log 概率
        actor_logps = (actor_token_logps * final_mask).sum(dim=-1)

        # 计算采样的 token 在 old_actor_model 和 reference_model 中的预测概率
        with torch.no_grad():
            old_logits = old_actor_model(
                input_ids=gen_out, attention_mask=full_mask
            ).logits
            old_actor_token_logps = (
                F.log_softmax(old_logits[:, :-1, :], dim=-1)
                .gather(dim=-1, index=labels.unsqueeze(-1))
                .squeeze(-1)
            )
            # old_actor_logps.shape = (B * num_gen, )
            old_actor_logps = (old_actor_token_logps * final_mask).sum(dim=1)

            ref_logits = ref_model(
                input_ids=gen_out, attention_mask=full_mask
            ).logits
            ref_token_logps = (
                F.log_softmax(ref_logits[:, :-1, :], dim=-1)
                .gather(dim=-1, index=labels.unsqueeze(-1))
                .squeeze(-1)
            )
            # ref_logps.shape = (B * num_gen, )
            ref_logps = (ref_token_logps * final_mask).sum(dim=1)

        # ========== 计算 PPO 损失 ==========
        # 1. policy loss
        # ratio.shape = (B * num_gen, )
        ratio = torch.exp(actor_logps - old_actor_logps)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, min=1.0 - args.clip_epsilon, max=1.0 + args.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 2. value loss
        value_loss = F.mse_loss(values, rewards)

        # PPO loss
        kl = (actor_logps - ref_logps).mean()
        loss = (policy_loss + args.vf_coef * value_loss + args.kl_coef * kl + aux_loss) / args.accumulation_steps
        loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            torch.nn.utils.clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            # 更新参数
            actor_optimizer.step()
            critic_optimizer.step()
            # 更新学习率
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        if step % args.update_old_actor_freq == 0:
            # 旧策略模型延迟更新, 用于模拟策略模型在若干 step 前的状态
            state_dict = (
                actor_model.module.state_dict()
                if isinstance(actor_model, DistributedDataParallel)
                else actor_model.state_dict()
            )
            old_actor_model.load_state_dict(
                {k: v.detach().cpu() for k, v in state_dict.items()}
            )
            old_actor_model.to(args.device)

        if is_main_process():
            # 统计 response 的平均长度
            lengths = full_mask[:, prompt_len:].sum(dim=-1)
            avg_len = lengths.float().mean()

            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            aux_loss_val = aux_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "actor_loss": actor_loss_val,
                    "critic_loss": critic_loss_val,
                    "aux_loss": aux_loss_val,
                    "reward": reward_val,
                    "kl": kl_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                })

            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Actor Loss: {actor_loss_val:.4f}, Critic Loss: {critic_loss_val:.4f}, Aux Loss: {aux_loss_val:.4f}, "
                   f"Reward: {reward_val:.4f}, KL: {kl_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            actor_model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_dim}{moe_suffix}.pth"

            if isinstance(actor_model, DistributedDataParallel):
                actor_state = actor_model.module.state_dict()
            else:
                actor_state = actor_model.state_dict()
            actor_state = {k: v.half().cpu() for k, v in actor_state.items()}
            torch.save(actor_state, ckp)

            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=actor_model,
                optimizer=actor_optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
                scheduler=actor_scheduler,
                critic_model=critic_model,
                critic_optimizer=critic_optimizer,
                critic_scheduler=critic_scheduler,
            )
            actor_model.train()
            del actor_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--save_weight", default="ppo_actor", type=str, help="保存权重的前缀名")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--max_seq_len", default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO裁剪参数（控制策略更新幅度）")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help="推理模型类型（0=普通模型，1=推理模型）")
    parser.add_argument("--update_old_actor_freq", type=int, default=4, help="更新old_actor_model的频率")
    parser.add_argument("--reward_model_path", type=str, default="../reward_model/internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
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
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 初始化模型 ==========
    base_weight = args.from_weight
    # 策略模型(Policy Model)
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # 旧策略模型
    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    old_actor_model.eval().requires_grad_(False)
    # 参考模型(Reference Model)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model.eval().requires_grad_(False)
    # 价值模型(Critic Model)
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{args.from_weight}_{lm_config.hidden_dim}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    # 奖励模型(Reward Model)
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)

    # 6. 加载数据集和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    actor_optimizer = torch.optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = torch.optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)

    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    # 这里就不手动调用 get_lr() 了
    # schedular 会在每个 step() 时修改 optimizer 中的 lr, lr 的最小值为 eta_min
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"cos_cached", "sin_cached"}
        critic_model._ddp_params_and_buffers_to_ignore = {"cos_cached", "sin_cached"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + start_step + 1, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step, wandb)
        else:  # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            ppo_train_epoch(epoch, loader, len(loader), ref_model, actor_scheduler, critic_scheduler, reward_model, 0, wandb)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()