import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import pandas as pd
import torch
import io
from tqdm import tqdm
import random
from PIL import Image as PILImage
import gc
from torch.multiprocessing import Process, Queue, set_start_method
import torch.multiprocessing as mp
from datasets import Dataset, Features, Image, Value
import pyarrow as pa
import pyarrow.parquet as pq

# Test Mode
TEST_MODE = False
TEST_SAMPLES = 16

# Positive magic suffix for better quality (from official tutorial)
POSITIVE_MAGIC_EN = ", Ultra HD, 4K, cinematic composition."

# Detailed prompt generation template for SCENE edits in off-road scenarios
PROMPT_GENERATION_TEMPLATE = """
For the scene edit instruction: "{edit_instruction}", generate **two** detailed English image-generation prompts for off-road scene-content changes:
1) a BEFORE-SCENE prompt that depicts the off-road scene before the scene change (original scene state),
2) an AFTER-SCENE prompt that depicts the off-road scene after the scene change (target scene state).

ATTENTION:
** Focus on off-road terrain and scene/texture/vegetation differences **
** Scene changes MUST be SIGNIFICANT and OBVIOUS with dramatic visual differences (e.g., dense grass -> sparse grass, smooth asphalt -> rough gravel) **
** Include terrain details and keep scene composition identical **

REQUIREMENTS (follow these exactly):
- Provide EXACTLY two prompts, in the exact format shown below. MUST be 30-50 words each.
- The two prompts must be **extremely similar**: OVER 90% of words should be identical to ensure scene continuity. Only the scene-related descriptors (vegetation density, surface texture, ground cover, debris) should differ significantly.
- Make scene/texture changes between the prompts **obvious and dramatic** visually, but keep all other scene elements identical (camera angle, background features, subject placement).
- Keep prompts DETAILED with specific terrain, surface texture, and composition descriptions.
- The BEFORE prompt must clearly describe the original scene state (e.g., "dense tall grass", "smooth wet asphalt").
- The AFTER prompt must clearly describe the target scene state (e.g., "sparse short grass", "coarse dry gravel").
- Include specific details: ground cover, surface roughness, vegetation density, visibility, camera angle, weather if relevant, and atmospheric effects.

FORMAT (must match exactly):
INPUT PROMPT: Generate image: [detailed description of BEFORE scene state in off-road scenario]
OUTPUT PROMPT: Generate image: [detailed description of AFTER scene state in off-road scenario]

Now generate the two prompts following the above rules.
"""

def load_vl_model(device_id):
    """Load VL model on specific device"""
    print(f"[GPU {device_id}] Loading Qwen2.5-VL...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    device = f'cuda:{device_id}'
    vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Path_to_Model//Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map=device
    )
    vl_processor = AutoProcessor.from_pretrained(
        "Path_to_Model/Qwen2.5-VL-7B-Instruct"
    )
    
    print(f"[GPU {device_id}] Qwen2.5-VL loaded successfully!")
    return vl_model, vl_processor

def load_image_model(device_id):
    """Load image generation model on specific device - following official tutorial"""
    print(f"[GPU {device_id}] Loading Qwen-Image model...")
    from diffusers import DiffusionPipeline
    
    device = f'cuda:{device_id}'
    
    try:
        # Use bfloat16 as recommended in official tutorial
        image_pipe = DiffusionPipeline.from_pretrained(
            "/mnt/bn/strategy-mllm-train/common/models/Qwen-Image",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        
        image_pipe = image_pipe.to(device)
        
        # Enable memory optimizations
        if hasattr(image_pipe, 'enable_vae_slicing'):
            image_pipe.enable_vae_slicing()
        
        if hasattr(image_pipe, 'enable_vae_tiling'):
            image_pipe.enable_vae_tiling()
            
        if hasattr(image_pipe, 'enable_attention_slicing'):
            image_pipe.enable_attention_slicing(1)
        
        print(f"[GPU {device_id}] Qwen-Image model loaded successfully!")
        return image_pipe
        
    except Exception as e:
        print(f"[GPU {device_id}] Error loading Qwen-Image model: {e}")
        raise

def generate_prompts_with_vl(vl_model, vl_processor, edit_instruction, device):
    """Use Qwen2.5-VL to generate two detailed prompts for scene edits"""
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_GENERATION_TEMPLATE.format(edit_instruction=edit_instruction)},
            ],
        }]
        
        text = vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = vl_processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            generated_ids = vl_model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.15,
                do_sample=True,
                top_p=0.9
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = vl_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        lines = [line.strip() for line in output_text.strip().split('\n') if line.strip()]
        input_prompt = ""
        output_prompt = ""
        
        for line in lines:
            if line.upper().startswith("INPUT PROMPT:"):
                input_prompt = line[len("INPUT PROMPT:"):].strip()
            elif line.upper().startswith("OUTPUT PROMPT:"):
                output_prompt = line[len("OUTPUT PROMPT:"):].strip()
        
        if input_prompt and not input_prompt.startswith("Generate image:"):
            input_prompt = "Generate image: " + input_prompt
        if output_prompt and not output_prompt.startswith("Generate image:"):
            output_prompt = "Generate image: " + output_prompt
        
        if (not input_prompt or not output_prompt or 
            len(input_prompt) < 30 or len(output_prompt) < 30 or 
            input_prompt == output_prompt):
            raise ValueError("Generated prompts are invalid or too similar")
        
        return input_prompt, output_prompt
        
    except Exception as e:
        print(f"Error parsing prompts for instruction '{edit_instruction}': {e}")
        return (
            f"Generate image: Off-road terrain scene, original scene state, {edit_instruction}",
            f"Generate image: Off-road terrain scene, changed scene state, {edit_instruction}"
        )

def pil_to_dict(image):
    """Convert PIL Image to dictionary format for HuggingFace datasets"""
    # Convert PIL image to bytes in a format that HF datasets can handle
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    return {
        'bytes': buffer.getvalue(),
        'path': None
    }

def worker_process(device_id, task_queue, result_queue, progress_queue):
    """Worker process for parallel processing"""
    try:
        # Load models on this GPU
        vl_model, vl_processor = load_vl_model(device_id)
        image_pipe = load_image_model(device_id)
        device = f'cuda:{device_id}'
        
        print(f"[GPU {device_id}] Worker ready, waiting for tasks...")
        
        while True:
            task = task_queue.get()
            
            if task is None:  # Poison pill
                print(f"[GPU {device_id}] Received poison pill, shutting down...")
                break
            
            instruction = task
            
            try:
                # Generate prompts using VL model
                input_prompt, output_prompt = generate_prompts_with_vl(
                    vl_model, vl_processor, instruction, device
                )
                
                # Clear cache after VL generation
                torch.cuda.empty_cache()
                
                # Remove "Generate image:" prefix for actual generation
                input_prompt_clean = input_prompt.replace("Generate image: ", "")
                output_prompt_clean = output_prompt.replace("Generate image: ", "")
                
                # Add positive magic suffix for better quality (official recommendation)
                input_prompt_final = input_prompt_clean + POSITIVE_MAGIC_EN
                output_prompt_final = output_prompt_clean + POSITIVE_MAGIC_EN
                
                # Use recommended aspect ratio and parameters from official tutorial
                # Using 1:1 aspect ratio for consistency
                width, height = 1328, 1328
                
                base_seed = random.randint(0, 1000000)
                
                # Parameters following official tutorial
                generation_params = {
                    "width": width,
                    "height": height,
                    "num_inference_steps": 50,  # Increased from 20 to 50
                    "true_cfg_scale": 4.0,  # Changed from guidance_scale to true_cfg_scale
                    "negative_prompt": " "  # Empty string as recommended
                }
                
                print(f"[GPU {device_id}] Generating input image with prompt: {input_prompt_clean[:80]}...")
                
                # Generate input image (before scene change)
                with torch.no_grad():
                    input_image = image_pipe(
                        prompt=input_prompt_final,
                        generator=torch.Generator(device=device).manual_seed(base_seed),
                        **generation_params
                    ).images[0]
                
                torch.cuda.empty_cache()
                
                print(f"[GPU {device_id}] Generating output image with prompt: {output_prompt_clean[:80]}...")
                
                # Generate output image (after scene change)
                with torch.no_grad():
                    output_image = image_pipe(
                        prompt=output_prompt_final,
                        generator=torch.Generator(device=device).manual_seed(base_seed + 1),
                        **generation_params
                    ).images[0]
                
                torch.cuda.empty_cache()
                
                # Convert to HF datasets compatible format
                result = {
                    'input_image': pil_to_dict(input_image),
                    'input_text': instruction,
                    'output_image': pil_to_dict(output_image)
                }
                
                result_queue.put(result)
                progress_queue.put(1)  # Signal progress
                
                print(f"[GPU {device_id}] Successfully processed: {instruction[:50]}...")
                
            except Exception as e:
                print(f"[GPU {device_id}] Error processing instruction '{instruction}': {e}")
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()
                gc.collect()
                progress_queue.put(1)  # Still update progress even on error
                continue
    
    except Exception as e:
        print(f"[GPU {device_id}] Worker process error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"[GPU {device_id}] Worker process terminated")

def save_checkpoint_hf_dataset(data_list, checkpoint_num):
    """Save checkpoint as HuggingFace dataset"""
    if not data_list:
        return
    
    try:
        # Define features with Image type for proper HF dataset handling
        features = Features({
            'input_image': Image(),
            'input_text': Value('string'),
            'output_image': Image()
        })
        
        # Create dataset from list
        dataset = Dataset.from_list(data_list, features=features)
        
        # Save as parquet with proper image encoding
        checkpoint_file = f"./checkpoint{checkpoint_num}_scene.parquet"
        dataset.to_parquet(checkpoint_file)
        
        print(f"HF Dataset Checkpoint {checkpoint_num} saved: {len(data_list)} samples")
    except Exception as e:
        print(f"Error saving checkpoint {checkpoint_num}: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Set multiprocessing start method
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Read instructions from scene.txt
    scene_file = 'Path_to_TerraBagel/data_generation/create_instruction/edit_instruction/scene.txt'
    with open(scene_file, 'r', encoding='utf-8') as f:
        instructions = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Loaded {len(instructions)} scene edit instructions")
    
    # Processing parameters
    num_gpus = 8
    target_count = TEST_SAMPLES if TEST_MODE else len(instructions)
    checkpoint_interval = 2 if TEST_MODE else 2000
    
    instructions_to_process = instructions[:target_count]
    print(f"Processing {target_count} instructions in {'TEST' if TEST_MODE else 'PRODUCTION'} mode")
    print(f"Using {num_gpus} GPUs for parallel processing")
    
    # Create queues for parallel processing
    task_queue = Queue()
    result_queue = Queue()
    progress_queue = Queue()
    
    # Add all tasks to queue
    for instruction in instructions_to_process:
        task_queue.put(instruction)
    
    # Add poison pills to stop workers
    for _ in range(num_gpus):
        task_queue.put(None)
    
    # Start worker processes
    processes = []
    for device_id in range(num_gpus):
        p = Process(target=worker_process, args=(device_id, task_queue, result_queue, progress_queue))
        p.start()
        processes.append(p)
        print(f"Started worker process for GPU {device_id}")
    
    # Collect results with overall progress bar
    data_list = []
    checkpoint_count = 0
    
    with tqdm(total=len(instructions_to_process), desc="Overall Progress", position=0) as pbar:
        completed = 0
        while completed < len(instructions_to_process):
            # Check for progress updates
            try:
                progress_queue.get(timeout=0.1)
                completed += 1
                pbar.update(1)
            except:
                pass
            
            # Collect results
            try:
                result = result_queue.get(timeout=0.1)
                data_list.append(result)
                
                # Save checkpoint periodically
                if len(data_list) % checkpoint_interval == 0:
                    checkpoint_count += 1
                    save_checkpoint_hf_dataset(data_list.copy(), checkpoint_count)
            except:
                pass
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    print("\nAll worker processes completed")
    
    # Collect any remaining results
    while not result_queue.empty():
        result = result_queue.get()
        data_list.append(result)
    
    # Save final checkpoint if needed
    if len(data_list) % checkpoint_interval != 0:
        checkpoint_count += 1
        save_checkpoint_hf_dataset(data_list, checkpoint_count)
    
    # Save final dataset as HuggingFace dataset
    if data_list:
        try:
            # Define features with Image type
            features = Features({
                'input_image': Image(),
                'input_text': Value('string'),
                'output_image': Image()
            })
            
            # Create dataset from list
            dataset = Dataset.from_list(data_list, features=features)
            
            # Save as parquet
            dataset.to_parquet("./scene.parquet")
            print(f"\nFinal scene.parquet saved: {len(data_list)} samples")
            
            # Also save as HuggingFace dataset directory for easier loading
            dataset.save_to_disk("./scene_dataset")
            print(f"Dataset also saved to ./scene_dataset directory")
            
            if TEST_MODE and data_list:
                sample = data_list[0]
                print(f"\nSample data:")
                print(f"Input text: {sample['input_text']}")
                print(f"Input image type: {type(sample['input_image'])}")
                print(f"Output image type: {type(sample['output_image'])}")
        except Exception as e:
            print(f"Error saving final dataset: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== Processing Complete ===")
    print(f"Total samples generated: {len(data_list)}")
    print(f"Checkpoints saved: {checkpoint_count}")

if __name__ == "__main__":
    main()
