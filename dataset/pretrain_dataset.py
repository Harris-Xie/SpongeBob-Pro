"""
SpongeBob 预训练数据集
加载预处理好的二进制数据
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """
    预训练数据集：加载.bin文件
    数据格式：(num_chunks, seq_len) 的 uint16 数组
    """
    def __init__(self, data_path, seq_len=512):
        """
        Args:
            data_path: .bin文件路径
            seq_len: 序列长度（用于验证，实际从.meta读取）
        """
        if not data_path.endswith('.bin'):
            data_path = data_path + '.bin'
        
        # 加载元信息
        meta_path = data_path.replace('.bin', '.meta')
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        
        # 验证序列长度
        assert self.meta['seq_len'] == seq_len, f"seq_len mismatch: {self.meta['seq_len']} vs {seq_len}"
        
        # 使用内存映射加载数据（不占用内存）
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r', shape=tuple(self.meta['shape']))
        
        print(f"Dataset loaded: {len(self)} chunks from {data_path}")
    
    def __len__(self):
        return self.meta['num_chunks']
    
    def __getitem__(self, idx):
        """
        返回 (input_ids, labels)
        注意：不做 shift，让模型自己处理
        input_ids 和 labels 是同一个序列，模型会在内部做 shift
        """
        chunk = torch.from_numpy(self.data[idx].astype(np.int64))
        # 直接返回整个 chunk 作为 input_ids 和 labels
        return chunk.clone(), chunk.clone()

        

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = conversations.copy()
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
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

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
