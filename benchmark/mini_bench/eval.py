# -*- coding: utf-8 -*-
"""Mini bench：推理 + Ollama Judge 打分"""
import json
import os
import re
import torch

_BENCH_JSONL = os.path.join(os.path.dirname(__file__), "100miniSponge.jsonl")
DIMENSIONS = ("fluency", "factuality", "instruction_following")

JUDGE_PROMPT = """你是一个严格的 0.1B 小模型评测员。请阅读用户的【问题】和模型的【回复】，从以下三个维度进行二元判定（0 或 1）。

【判定标准】
1. **fluency (语句流畅性)**: 1=通顺可读，0=乱码/复读机/严重截断
2. **factuality (事实准确性)**: 1=基本符合事实，0=严重幻觉/完全错误
3. **instruction_following (指令遵循)**: 1=回答了问题，0=答非所问

【待测数据】
User: {question}
Model: {response}

仅输出 JSON：{{"fluency": 0 or 1, "factuality": 0 or 1, "instruction_following": 0 or 1}}"""


def run_inference(model, tokenizer, device=None, num_samples=3, max_prompts=None, batch_size=20):
    """批量推理：每批 batch_size 条 prompt × num_samples 次生成"""
    with open(_BENCH_JSONL, "r", encoding="utf-8") as f:
        prompts = [json.loads(l)["prompt"] for l in f]
    if max_prompts:
        prompts = prompts[:max_prompts]
    
    all_pairs = []
    for batch_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_idx:batch_idx + batch_size]
        print(f"  [eval] 推理 {batch_idx+1}-{min(batch_idx+len(batch_prompts), len(prompts))}/{len(prompts)} 条...")
        
        all_convs = [[{"role": "user", "content": p}] for p in batch_prompts]
        all_txts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in all_convs]
        inputs = tokenizer(all_txts, padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"], attention_mask=inputs["attention_mask"],
                num_return_sequences=num_samples, max_new_tokens=512, 
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        for i, prompt in enumerate(batch_prompts):
            original_len = (inputs["attention_mask"][i] == 1).sum().item()
            rs = [tokenizer.decode(outputs[i*num_samples+j][original_len:], skip_special_tokens=False).strip() 
                  for j in range(num_samples)]
            all_pairs.append((prompt, rs))
    
    return all_pairs


def _parse_judge_json(text):
    """从 Judge 输出提取 JSON 并解析三个维度"""
    for pattern in [r"```(?:json)?\s*(\{[^`]*\})\s*```", r"(\{[^{}]*\})"]:
        for raw in re.findall(pattern, text, re.DOTALL | re.IGNORECASE):
            try:
                d = json.loads(raw.strip())
                result = {k: 1 if d.get(k, d.get(k.replace("_", " "), 0)) >= 1 else 0 for k in DIMENSIONS}
                if len(result) == 3:
                    return result
            except:
                continue
    return None


def _judge_one(prompt, response, ollama_url, ollama_model):
    """单次 Judge 请求"""
    import requests
    try:
        r = requests.post(ollama_url.rstrip("/") + "/api/chat", json={
            "model": ollama_model,
            "messages": [{"role": "user", "content": JUDGE_PROMPT.format(question=prompt, response=response)}],
            "stream": False
        }, timeout=120)
        r.raise_for_status()
        out = (r.json().get("message") or {}).get("content", "") or r.json().get("response", "")
        return _parse_judge_json(out), None
    except Exception as e:
        return None, str(e)[:200]


def run_judge(pairs, ollama_url="http://127.0.0.1:11434", ollama_model="qwen3:1.7b", return_details=False, max_workers=10):
    """Judge 评测（同步版本）"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    tasks = [(i, j, p, r) for i, (p, rs) in enumerate(pairs) for j, r in enumerate(rs)]
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(_judge_one, p, r, ollama_url, ollama_model): (i, j) 
                         for i, j, p, r in tasks}
        for future in as_completed(future_to_idx):
            i, j = future_to_idx[future]
            results[(i, j)] = future.result()[0]
    
    dim_data = {d: {"scores": [], "pass_per_prompt": []} for d in DIMENSIONS}
    details = []
    
    for i, (prompt, responses) in enumerate(pairs):
        passed_any = {d: False for d in DIMENSIONS}
        judge_results = []
        for j in range(len(responses)):
            parsed = results.get((i, j))
            judge_results.append(parsed)
            if parsed:
                for d in DIMENSIONS:
                    v = parsed.get(d, 0)
                    dim_data[d]["scores"].append(float(v))
                    if v == 1:
                        passed_any[d] = True
        for d in DIMENSIONS:
            dim_data[d]["pass_per_prompt"].append(passed_any[d])
        if return_details:
            details.append({"prompt": prompt, "responses": responses, 
                           "judge_results": judge_results, "pass_any": passed_any})
    
    metrics = {}
    for d in DIMENSIONS:
        scores = dim_data[d]["scores"]
        pass_pp = dim_data[d]["pass_per_prompt"]
        metrics[f"{d}_avg3"] = sum(scores) / len(scores) if scores else 0.0
        metrics[f"{d}_pass3"] = sum(pass_pp) / len(pairs) if pairs else 0.0
    metrics["mean_avg3"] = sum(metrics[f"{d}_avg3"] for d in DIMENSIONS) / 3
    metrics["mean_pass3"] = sum(metrics[f"{d}_pass3"] for d in DIMENSIONS) / 3
    
    return (metrics, details) if return_details else metrics


def run_judge_async(pairs, ollama_url="http://127.0.0.1:11434", ollama_model="qwen3:1.7b",
                     output_file=None, swanlab_log_fn=None, global_step=None, max_workers=10):
    """异步 Judge：后台执行，不阻塞训练"""
    import threading
    
    def _background():
        try:
            metrics, details = run_judge(pairs, ollama_url, ollama_model, return_details=True, max_workers=max_workers)
            if output_file:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    for item in details:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            if swanlab_log_fn:
                swanlab_log_fn({f"eval/{k}": v for k, v in metrics.items()}, step=global_step)
            print(f"  [judge] step={global_step} fluency={metrics['fluency_avg3']:.3f}/{metrics['fluency_pass3']:.3f} "
                  f"factuality={metrics['factuality_avg3']:.3f}/{metrics['factuality_pass3']:.3f} "
                  f"if={metrics['instruction_following_avg3']:.3f}/{metrics['instruction_following_pass3']:.3f} "
                  f"mean={metrics['mean_avg3']:.3f}/{metrics['mean_pass3']:.3f}")
        except Exception as e:
            print(f"  [judge] 失败: {e}")
    
    threading.Thread(target=_background, daemon=True).start()
    print(f"  [judge] 后台启动，训练继续...")


