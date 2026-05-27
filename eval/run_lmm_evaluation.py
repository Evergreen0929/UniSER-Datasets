import os
os.environ.pop("LOCAL_RANK", None)
import torch
import pandas as pd
from pathlib import Path
import argparse
import re
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct" 
BASE_DIR = Path.cwd()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("Warning: Running on CPU is extremely slow. A powerful GPU is required for this model.")

# --- Model initialization ---
def initialize_lmm(model_id):
    """Load and initialize the LMM model and its processor."""
    print(f"Loading model and processor: {model_id}. This is a very large model and will take several minutes...")
    # Note: per the earlier error log, loading Qwen2.5-VL required
    # AutoModelForCausalLM. If an AutoClass-related ValueError appears again,
    # switch back from AutoModelForVision2Seq.

    # FIX: Use AutoModelForVision2Seq for this multimodal model.
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype="auto",
        # device_map="auto",
        device_map={"": 0},
        load_in_4bit=True,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("Model and processor loaded successfully.")
    return model, processor

# --- Core evaluation logic ---
def parse_percentage_score(text: str) -> float:
    """Parse a percentage score from an LMM reply (formats like '评分：85%' or 'Score: 85%')."""
    # Matches both the Chinese keyword used in the prompt ('评分') and an English fallback.
    match = re.search(r'(?:评分|Score)\s*[:：]\s*(\d+\.?\d*)\s*%?', text, re.IGNORECASE)
    if match:
        score = float(match.group(1))
        if 0 <= score <= 100:
            return score
    return -1.0

def evaluate_pair_with_lmm(model, processor, input_img_path, method_img_path, task_name):
    """Evaluate an image pair with the LMM using high-intensity prompts."""
    try:
        # --- MODIFICATION START ---
        # 1. Load the original input image to get its resolution
        input_image = Image.open(input_img_path).convert("RGB")
        target_size = input_image.size  # Get the target resolution (width, height)

        # 2. Load the prediction image from the method
        method_image = Image.open(method_img_path).convert("RGB")
        original_size = method_image.size

        # 3. Resize the prediction image to match the input if resolutions differ
        if original_size != target_size:
            print(f"   - Resizing prediction from {original_size} to {target_size} to match input.")
            # Image.Resampling.LANCZOS is a high-quality filter for resizing.
            # For older Pillow versions, you might need to use Image.LANCZOS
            method_image = method_image.resize(target_size, Image.Resampling.LANCZOS)
        # --- MODIFICATION END ---

    except Exception as e:
        # Updated error message to include potential resizing errors
        print(f"Error opening or resizing images: {e}")
        return -1.0

    artifact_map = {
        'haze': '雾气',
        'shadow': '阴影',
        'reflection': '反射干扰',
        'lens_flares': '镜头眩光'
    }
    artifact_name = artifact_map.get(task_name, '瑕疵')

    question = (f"你是一位顶级的图像质量评估专家。你的任务是比较两张图片并给出量化评估。\n"
                f"图A是带有'{artifact_name}'的原始图像。图B是经过算法处理后的图像。\n"
                f"你的唯一目标是评估图B相比图A'{artifact_name}'减弱的百分比。\n"
                f"你的回答必须且只能包含最终的百分比，并严格遵循'评分：[数字]%'的格式。不要提供任何描述、解释或其他无关文字。")

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": question}]}]

    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=prompt,
        images=[input_image, method_image], # Use the potentially resized method_image
        return_tensors="pt"
    ).to(DEVICE)

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    try:
        response_text = response.split(question)[-1].strip()
    except IndexError:
        response_text = response

    print(f"   - LMM Raw Response: {response_text}")
    return parse_percentage_score(response_text)


# --- File handling and main entry point ---
def find_corresponding_input(method_file: Path, input_dir: Path) -> Path | None:
    match = re.search(r'(\d+)', method_file.name)
    if not match: return None
    file_id = match.group(1)
    search_pattern = f"row_{file_id}_*"
    found_files = list(input_dir.glob(search_pattern))
    return found_files[0] if found_files else None

def main():
    parser = argparse.ArgumentParser(description="Run LMM-based percentage removal evaluation using Qwen2.5-VL.")
    parser.add_argument('--tasks', required=True, nargs='+', help='One or more task directories to evaluate (e.g., shadow haze).')
    args = parser.parse_args()
    task_names = args.tasks

    model, processor = initialize_lmm(MODEL_ID)
    all_results = []
    
    for task_name in task_names:
        task_path = BASE_DIR / task_name
        if not task_path.is_dir(): continue
        
        input_dir = task_path / 'input'
        if not input_dir.is_dir(): continue

        methods = [d.name for d in task_path.iterdir() if d.is_dir() and d.name != 'input']
        
        for method_name in methods:
            print(f"\nEvaluating Task: [{task_name}], Method: [{method_name}]")
            method_path = task_path / method_name
            
            image_files = sorted(list(method_path.glob('*.png')) + list(method_path.glob('*.jpg')))
            for method_img_path in image_files:
                input_img_path = find_corresponding_input(method_img_path, input_dir)
                if input_img_path:
                    print(f"Processing pair: [Input: {input_img_path.name}] vs [Method: {method_img_path.name}]")
                    score = evaluate_pair_with_lmm(model, processor, input_img_path, method_img_path, task_name)
                    if score != -1.0:
                        all_results.append({
                            "task": task_name,
                            "method": method_name,
                            "image_pair": f"{input_img_path.name} vs {method_img_path.name}",
                            "Removal_Percentage(H)": score
                        })
                else:
                    print(f"  - Could not find matching input for {method_img_path.name}")

    if not all_results:
        print("No results were generated.")
        return

    # --- Aggregate and save results ---
    results_df = pd.DataFrame(all_results)
    detailed_csv_path = BASE_DIR / "evaluation_detailed_results_qwen2.5vl-72b.csv"
    results_df.to_csv(detailed_csv_path, index=False)
    print(f"\nDetailed Qwen2.5-VL-72B results saved to: {detailed_csv_path}")

    summary_df = results_df.groupby(['task', 'method'])['Removal_Percentage(H)'].mean().reset_index()
    
    summary_excel_path = BASE_DIR / "evaluation_summary_results_qwen2.5vl-72b.xlsx"
    summary_df.to_excel(summary_excel_path, index=False)
    print(f"Summary Qwen2.5-VL-72B results saved to: {summary_excel_path}")

    print("\n--- Qwen2.5-VL-72B Evaluation Summary (Removal %) ---")
    print(summary_df.to_string())

if __name__ == "__main__":
    main()