## 📌 原代码链接

本项目基于minimind复刻，原作者项目链接: https://github.com/jingyaogong/minimind

## 📌 快速开始

### Ⅰ 安装依赖

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

### Ⅱ 下载数据集

从链接(https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) 下载所需数据文件，并放入 `./dataset` 目录

- 预训练: pretrain_t2t.jsonl
- 有监督微调: sft_t2t.jsonl
- RL: rlaif.jsonl

### Ⅲ 开始训练

- 预训练: python train_pretrain.py --epochs 3 --use_moe 1 --num_hidden_layers 6 --hidden_size 640
- SFT: python train_full_sft.py --epochs 2 --use_moe 1 --num_hidden_layers 6 --hidden_size 640 --accumulation_steps 4 --batch_size 8
- PPO: python train_ppo.py --epochs 1 --use_moe 1 --num_hidden_layers 6 --hidden_size 640 --accumulation_steps 1 --batch_size 2
- GRPO: python train_grpo.py --epochs 2 --use_moe 1 --num_hidden_layers 6 --hidden_size 640 --accumulation_steps 1 --batch_size 2
