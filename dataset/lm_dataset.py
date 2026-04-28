import json
import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset, Features, Sequence, Value


def pre_processing_chat(conversations, add_system_ratio: float = 0.2):
    """
    为了让模型适应不同格式的输入, 有 add_system_ratio 的概率再每条数据的开始加入系统提示词
    """

    # 不对含有 tools 的问答做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio: float = 0.2):
    """
    为了让模型既见过带 <think> 格式的数据, 也防止被空 <think> 块过度的污染, 对文本中空 <think> 块做概率删除
    """
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = self.tokenizer(
            sample['text'],
            max_length=self.max_length - 2,
            add_special_tokens=False,
            truncation=True,
        )["input_ids"]
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length: int = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features(
            {'conversations': [
                {
                    'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')
                }
            ]}
        )
        self.bos_id = self.tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = self.tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=data_path, split='train', features=features)

    def create_chat_template(self, conversations):
        """
        为了将结构化的问答对拼接成纯文本字符串, 如数据集中的原始格式为:

        {
            "conversations": [
                {"content": "给定一段文本和关键词列表，删除文本中包含所有给定关键词的子字符串。\n文本：\"这是一个测试句子，目的是看看模型是否可以正确地从这个句子中删除关键词。\"\\n关键词列表：[‘测试’，‘模型’]", "role": "user"},
                {"content": "删除包含所有给定关键词的子字符串后，文本变为：\"这是一个句子，目的是看看是否可以正确地从这个句子中删除关键词。\"", "role": "assistant"},
                {"content": "好的。现在请你将这个文本中的所有的逗号都替换成空格。", "role": "user"},
                {"content": "好的，请稍等一下，现在我会将文本中的所有逗号替换为空格。处理后文本为：\"这是一个句子 目的是看看是否可以正确地从这个句子中删除关键词。\"。处理结果如何？", "role": "assistant"}
            ]
        }

        经过拼接后的到纯文本字符串:

        <|im_start|>system
        You are a helpful assistant<|im_end|>
        <|im_start|>user
        给定一段文本和关键词列表，删除文本中包含所有给定关键词的子字符串。
        文本："这是一个测试句子，目的是看看模型是否可以正确地从这个句子中删除关键词。"\n关键词列表：[‘测试’，‘模型’]<|im_end|>
        <|im_start|>assistant
        删除包含所有给定关键词的子字符串后，文本变为："这是一个句子，目的是看看是否可以正确地从这个句子中删除关键词。"<|im_end|>
        <|im_start|>user
        好的。现在请你将这个文本中的所有的逗号都替换成空格。<|im_end|>
        <|im_start|>assistant
        好的，请稍等一下，现在我会将文本中的所有逗号替换为空格。处理后文本为："这是一个句子 目的是看看是否可以正确地从这个句子中删除关键词。"。处理结果如何？<|im_end|>
        """
        messages = []
        tools = None
        for conv in conversations:
            conv = dict(conv)
            # 单独处理 tools, 将其转化为 json 格式
            if conv.get('role') == 'system' and conv.get('tools'):
                if isinstance(conv['tools'], str):
                    tools = json.loads(conv['tools'])
                else:
                    tools = conv['tools']
            # 单独处理 tool_calls, 将其转化为 json 格式
            if conv.get('tool_calls'):
                if isinstance(conv['tool_calls'], str):
                    conv['tool_calls'] = json.loads(conv['tool_calls'])
            messages.append(conv)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # 不会在末尾加入 <|im_start|>assistant 标记
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]['conversations']
        conversations = pre_processing_chat(sample)
        prompt = self.create_chat_template(conversations)
        prompt = post_processing_chat(prompt)

        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return input_ids, labels


class RLAIFDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, thinking_ratio: float = 0.2):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ''
        }


if __name__ == '__main__':
    """ 测试 SFTDataset """
    # dataset = SFTDataset(
    #     data_path='sft_t2t.jsonl',
    #     tokenizer=AutoTokenizer.from_pretrained('../model'),
    # )
    # print(dataset[0])

    """ 测试 RLAIFDataset """
    dataset = RLAIFDataset(
        data_path='rlaif.jsonl',
        tokenizer=AutoTokenizer.from_pretrained('../model'),
    )
    print(dataset[0])
