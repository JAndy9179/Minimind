## 📌 连接服务器

Minimind-small

scp -P 40296 -r E:\python\project\Minimind-2\reward_model root@connect.nmb2.seetacloud.com:/root/autodl-tmp/Minimind

scp -P 40296 -r root@connect.nmb2.seetacloud.com:/root/autodl-tmp/Minimind/checkpoints E:\python\project\Minimind-2

scp -P 40296 -r root@connect.nmb2.seetacloud.com:/root/autodl-tmp/Minimind/out E:\python\project\Minimind-2

Minimind-Large

scp -P 49386 -r E:\python\project\Minimind-2 root@connect.bjb1.seetacloud.com:/root/autodl-tmp

scp -P 49386 -r root@connect.bjb1.seetacloud.com:/root/autodl-tmp/Minimind/out E:\python\project\Minimind-2

scp -P 49386 -r root@connect.bjb1.seetacloud.com:/root/autodl-tmp/Minimind/out E:\python\project\Minimind-2

## 📌 创建虚拟环境

conda remove --name minimind --all

conda create --name minimind python=3.10

conda activate minimind

conda deactivate

** 安装 tmux: apt update && apt install tmux -y

## 📌 安装依赖

查看以缓存的依赖: pip cache dir

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

pip uninstall -y torch torchvision torchaudio

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

## 📌 快捷指令

登录 Swanlab:
swanlab login
key: j5nvR2jtS44H9bUN9jBQ2

训练分词器: python train_tokenizer.py

预训练: 
Minimind-small: python train_pretrain.py --epochs 3 --use_moe 1 --num_hidden_layers 6 --hidden_size 640
Minimind-Large: torchrun --nproc_per_node 2 train_pretrain.py --epochs 5 --use_moe 1 --num_hidden_layers 12 --hidden_size 1536 --learning_rate 1.6e-4 --batch_size 16 --accumulation_steps 16  --use_wandb

SFT:
Minimind-small: python train_full_sft.py --epochs 2 --use_moe 1 --num_hidden_layers 6 --hidden_size 640 --accumulation_steps 4 --batch_size 8
Minimind-Large: 

PPO:
Minimind-small: python train_ppo.py --epochs 1 --use_moe 1 --num_hidden_layers 6 --hidden_size 640 --accumulation_steps 1 --batch_size 2
Minimind-Large: 

GRPO:
Minimind-small: python train_grpo.py --epochs 2 --use_moe 1 --num_hidden_layers 6 --hidden_size 640 --accumulation_steps 1 --batch_size 2
Minimind-Large: 

测试:
Minimind-small: 
python eval_llm.py --load_from ./model --weight pretrain --use_moe 1 --num_hidden_layers 6 --hidden_size 640
python eval_llm.py --load_from ./model --weight full_sft --use_moe 1 --num_hidden_layers 6 --hidden_size 640
python eval_llm.py --load_from ./model --weight ppo_actor --use_moe 1 --num_hidden_layers 6 --hidden_size 640

Minimind-Large: 
python eval_llm.py --load_from ./model --weight pretrain --use_moe 1 --num_hidden_layers 12 --hidden_size 1536
