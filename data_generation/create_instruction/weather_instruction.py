import os
import torch
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from torch.multiprocessing import Process, Queue, set_start_method, Manager
from pathlib import Path
import random
import time
import threading
from queue import Empty

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_instructions(gpu_id, num_batches, output_queue, model_path, batch_size=100):
    """
    在指定GPU上生成edit instructions，每生成batch_size条就发送一次
    """
    try:
        # 设置当前进程使用的GPU
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        
        # 为每个GPU设置不同的随机种子以增加多样性
        set_seed(42 + gpu_id)
        
        print(f"GPU {gpu_id}: Loading model...")
        
        # 加载模型到指定GPU
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": gpu_id},
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print(f"GPU {gpu_id}: Model loaded successfully, starting generation...")
        
        batch_instructions = []
        total_generated = 0
        
        for batch_idx in range(num_batches):
            prompt = """Generate EXACT 10 simple and clear image edit instructions for off-road robot scenes. Each instruction should be on a separate line.

Requirements:
- Focus on GROUND-LEVEL terrain suitable for vehicle/robot navigation. MUST Avoid steep mountains, cliffs, peaks, vertical terrain, hills......
- Should be terra scenes. Avoid urban street, indoor......
- Transform a normal weather scene to extreme weather
- Be concise and direct (10-40 words maximum per instruction)
- Use simple language
- MUST be in English only
- Each instruction MUST use DIFFERENT terrain types
- Each instruction MUST use DIFFERENT extreme weather conditions

Good examples:
"Transform this muddy country road with fallen branches into a post-hailstorm scene with ice pellets covering the ground."
"Convert this rocky desert path featuring sparse vegetation into an intense sandstorm setting with limited visibility."
"Change this forest trail with wet leaves into a heavy thunderstorm environment with lightning and flooding."

Now generate 10 instructions in this style. Output ONLY the instructions, one per line, without numbering, quotes, or any other text:"""

            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # 使用tokenizer的chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 直接tokenize文本
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True
            )
            inputs = inputs.to(device)
            
            # 生成（增加max_new_tokens以容纳10个指令）
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=800,  # 增加以容纳10个指令
                    temperature=0.85,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # 解析输出：按行分割
            raw_output = output_text[0].strip()
            lines = raw_output.split('\n')
            
            valid_count = 0
            for line in lines:
                instruction = line.strip()
                
                # 跳过空行
                if not instruction:
                    continue
                
                # 清理输出：移除可能的序号、引号等
                instruction = instruction.replace('"', '').replace("'", "")
                
                # 移除可能的序号前缀（如 "1. ", "1) ", "- "等）
                instruction = instruction.lstrip('0123456789.-) ')
                
                # 再次去除首尾空格
                instruction = instruction.strip()
                
                # 跳过太短的行
                if len(instruction) < 10:
                    continue
                
                # 检查是否包含中文
                if any('\u4e00' <= char <= '\u9fff' for char in instruction):
                    continue
                
                # 检查长度是否合理（词数）
                word_count = len(instruction.split())
                if word_count < 5 or word_count > 60:
                    continue
                
                # 跳过明显不是指令的行（如包含"example", "instruction"等元文本）
                lower_inst = instruction.lower()
                if any(skip_word in lower_inst for skip_word in ['example:', 'instruction:', 'note:', 'format:']):
                    continue
                
                batch_instructions.append(instruction)
                total_generated += 1
                valid_count += 1
                
                # 达到10个就停止（避免解析过多）
                if valid_count >= 10:
                    break
            
            # 每生成batch_size条就发送一次
            if len(batch_instructions) >= batch_size:
                print(f"GPU {gpu_id}: Sending batch of {len(batch_instructions)} instructions (total: {total_generated})")
                try:
                    output_queue.put((gpu_id, batch_instructions.copy()), block=False)
                    batch_instructions.clear()
                except:
                    print(f"GPU {gpu_id}: Queue full, waiting...")
                    output_queue.put((gpu_id, batch_instructions.copy()), block=True, timeout=30)
                    batch_instructions.clear()
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"GPU {gpu_id}: Progress {batch_idx + 1}/{num_batches}, generated {total_generated} valid instructions")
        
        # 发送剩余的指令
        if batch_instructions:
            print(f"GPU {gpu_id}: Sending final batch of {len(batch_instructions)} instructions")
            output_queue.put((gpu_id, batch_instructions), block=True, timeout=30)
        
        # 发送完成信号
        output_queue.put((gpu_id, "DONE"), block=True, timeout=30)
        print(f"GPU {gpu_id}: Completed with {total_generated} valid instructions")
        
    except Exception as e:
        print(f"GPU {gpu_id}: Error occurred - {str(e)}")
        import traceback
        traceback.print_exc()
        try:
            output_queue.put((gpu_id, "DONE"), block=False)
        except:
            pass

def result_collector(output_queue, output_file, num_gpus):
    """
    后台线程，持续从队列中读取结果并实时保存到文件
    """
    completed_gpus = set()
    total_instructions = 0
    last_activity = time.time()
    no_data_timeout = 300  # 5分钟无数据才超时
    
    print("Collector: Started and waiting for data...")
    
    # 打开文件用于追加写入
    with open(output_file, 'w', encoding='utf-8') as f:
        while len(completed_gpus) < num_gpus:
            try:
                # 使用较短的超时，但会重试
                gpu_id, instructions = output_queue.get(timeout=5)
                last_activity = time.time()
                
                # "DONE"表示该GPU已完成
                if instructions == "DONE":
                    completed_gpus.add(gpu_id)
                    print(f"Collector: GPU {gpu_id} completed. ({len(completed_gpus)}/{num_gpus} GPUs finished)")
                    continue
                
                # 实时写入文件
                if isinstance(instructions, list):
                    for instruction in instructions:
                        f.write(f"{instruction}\n")
                        total_instructions += 1
                    
                    f.flush()  # 立即刷新到磁盘
                    print(f"Collector: Saved {len(instructions)} instructions from GPU {gpu_id} (total: {total_instructions})")
                
            except Empty:
                # 超时但继续等待
                elapsed = time.time() - last_activity
                if elapsed > no_data_timeout:
                    print(f"Collector: No data received for {no_data_timeout}s, stopping...")
                    break
                # 只是超时，继续等待
                continue
            except Exception as e:
                print(f"Collector: Error - {str(e)}")
                import traceback
                traceback.print_exc()
                # 不要立即退出，继续尝试
                continue
    
    print(f"Collector: Finished. Total {total_instructions} instructions saved to {output_file}")
    return total_instructions

def main():
    # 设置多进程启动方法
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # 配置参数
    model_path = "Path_to_Model//Qwen2.5-VL-7B-Instruct"
    num_gpus = 8
    total_instructions = 15000  # 总目标数量
    # 由于每次生成10个，所以需要的batch数量是 15000 / 10 / 8
    instructions_per_gpu = total_instructions // num_gpus  # 每个GPU生成1875个
    num_batches_per_gpu = instructions_per_gpu // 10  # 每个GPU需要188次生成（每次10个）
    batch_size = 100  # 每100条保存一次
    
    # 创建输出目录
    output_dir = Path("edit_instruction")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "weather.txt"
    
    # 如果文件已存在，先删除
    if output_file.exists():
        output_file.unlink()
        print(f"Removed existing file: {output_file}")
    
    print(f"Starting generation of {total_instructions} instructions across {num_gpus} GPUs")
    print(f"Each GPU will generate ~{instructions_per_gpu} instructions in {num_batches_per_gpu} batches (10 per batch)")
    print(f"Saving to file every {batch_size} instructions per GPU")
    print(f"Instructions will be concise, clear, and in English only\n")
    
    # 创建队列用于收集结果（使用Manager创建共享队列）
    manager = Manager()
    output_queue = manager.Queue(maxsize=500)
    
    # 创建并启动进程（先启动进程）
    processes = []
    for gpu_id in range(num_gpus):
        p = Process(
            target=generate_instructions,
            args=(gpu_id, num_batches_per_gpu, output_queue, model_path, batch_size)
        )
        p.start()
        processes.append(p)
        print(f"Started process on GPU {gpu_id}")
        time.sleep(0.5)  # 稍微延迟避免同时加载模型
    
    print("\nAll GPU processes started. Waiting for models to load...\n")
    time.sleep(5)  # 给GPU进程一些时间加载模型
    
    # 启动结果收集线程
    collector_thread = threading.Thread(
        target=result_collector,
        args=(output_queue, output_file, num_gpus),
        daemon=False
    )
    collector_thread.start()
    print("Started result collector thread\n")
    print(f"Results are being saved in real-time to: {output_file}\n")
    
    # 等待所有进程完成
    for idx, p in enumerate(processes):
        p.join()
        print(f"GPU {idx} process joined")
    
    print("\nAll GPU processes completed. Waiting for collector thread to finish...")
    
    # 等待收集线程完成（最多等待60秒）
    collector_thread.join(timeout=60)
    
    if collector_thread.is_alive():
        print("Warning: Collector thread still running, but continuing...")
    
    print("\nAll tasks completed. Analyzing results...")
    
    # 读取并分析最终结果
    if not output_file.exists():
        print("\n" + "="*60)
        print("ERROR: Output file was not created!")
        print("="*60)
        return
    
    with open(output_file, 'r', encoding='utf-8') as f:
        final_instructions = [line.strip() for line in f if line.strip()]
    
    # 检查是否有生成的指令
    if len(final_instructions) == 0:
        print("\n" + "="*60)
        print("ERROR: No valid instructions were generated!")
        print("="*60)
        return
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"Generation Complete!")
    print(f"{'='*60}")
    print(f"Total instructions generated: {len(final_instructions)}")
    print(f"Output file: {output_file}")
    
    # 计算平均长度（词数和字符数）
    avg_words = sum(len(inst.split()) for inst in final_instructions) / len(final_instructions)
    avg_chars = sum(len(inst) for inst in final_instructions) / len(final_instructions)
    print(f"Average instruction length: {avg_words:.1f} words, {avg_chars:.1f} characters")
    
    # 显示前15条样例
    print(f"\n{'='*60}")
    print("Sample Instructions (first 15):")
    print(f"{'='*60}")
    for i in range(min(15, len(final_instructions))):
        word_count = len(final_instructions[i].split())
        print(f"\n[Sample {i+1}] ({word_count} words)")
        print(f"{final_instructions[i]}")
    
    print(f"\n{'='*60}")
    print("Generation pipeline completed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()