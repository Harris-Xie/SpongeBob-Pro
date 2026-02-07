"""
SpongeBob 模型交互式对话脚本（简化版）
"""
import argparse
import torch
from transformers import AutoTokenizer, TextStreamer
from model.config import SpongeBobConfig
from model.model_spongebob_pro import SpongeBobForCausalLM

def main():
    parser = argparse.ArgumentParser(description="SpongeBob模型交互对话")
    parser.add_argument('--model_path', default='/apdcephfs_qy4/share_302593112/huaibingxie/SpongeBob/out_sft/exp_thinking/global_step_782/sft_768.pth', type=str, help="模型权重路径（.pth文件）")
    parser.add_argument('--tokenizer_path', default='./tokenizer_15k', type=str, help="Tokenizer路径")
    parser.add_argument('--model_type', default='sft', type=str, choices=['pretrain', 'sft'], help="模型类型：pretrain（文本续写）或 sft（对话）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=12, type=int, help="隐藏层数量")
    parser.add_argument('--max_new_tokens', default=2048, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.7, type=float, help="生成温度（0-1）")
    parser.add_argument('--top_p', default=0.7, type=float, help="nucleus采样阈值")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    args = parser.parse_args()
    
    # 自动推断模型类型（从文件名）
    if 'pretrain' in args.model_path:
        args.model_type = 'pretrain'
    elif 'sft' in args.model_path:
        args.model_type = 'sft'
    
    # 加载模型和tokenizer
    print(f'加载模型: {args.model_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    model = SpongeBobForCausalLM(SpongeBobConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers
    ))
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval().to(args.device)
    
    print(f'✅ 模型加载完成！设备: {args.device}')
    print(f'📝 模型类型: {args.model_type} ({"对话模式" if args.model_type == "sft" else "文本续写"})\n')
    print('='*60)
    print('💬 开始对话 (输入 exit 退出)')
    print('='*60)
    
    # 对话循环
    conversation = []
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
    
    while True:
        user_input = input('\n👤 你: ').strip()
        
        if user_input.lower() in ['exit', 'quit', '退出']:
            print('👋 再见！')
            break
        
        if not user_input:
            continue
        
        # 根据模型类型格式化输入
        if args.model_type == 'pretrain':
            # 预训练模型：直接文本续写，不保留历史
            formatted_input = user_input
            conversation = []  # 清空历史
        else:
            # SFT模型：使用 chat template，保留历史对话
            conversation.append({"role": "user", "content": user_input})
            formatted_input = tokenizer.apply_chat_template(
                conversation=conversation, 
                tokenize=False, 
                add_generation_prompt=True
            )
        
        inputs = tokenizer(formatted_input, return_tensors="pt").to(args.device)
        
        # 生成回复
        print('🧽 SpongeBob: ', end='', flush=True)
        with torch.no_grad():
            generated_ids = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=1.2
            )
        
        # 解码回复并添加到历史
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):], 
            skip_special_tokens=False
        )
        
        # 只有 SFT 模型才保留对话历史
        if args.model_type == 'sft':
            conversation.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
