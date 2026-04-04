"""
Dataset Preparation Script for Mini-R1
Prepares Countdown Game dataset with R1-style prompts

데이터셋 설명:
------------
이 스크립트는 Countdown Game 데이터셋을 R1 학습 형식으로 변환합니다.

원본 데이터 형태:
  {
    "nums": [19, 36, 55, 7],      # 사용 가능한 숫자들
    "target": 65                   # 목표 숫자
  }

변환 후 R1 형식:
  {
    "prompt": "시스템 메시지 + 사용자 질문 + <think>",  # 모델 입력
    "target": 65,                                      # 목표 숫자
    "nums": [19, 36, 55, 7]                            # 숫자 리스트
  }

모델이 생성해야 하는 출력:
  "추론 과정... </think>
  <answer> 55 + 36 - 7 - 19 </answer>"

사용법:
------
  python scripts/dataset_prep.py --num_samples 5000
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import yaml


def generate_r1_prompt(numbers: list, target: int, tokenizer) -> dict:
    """
    Generate R1-style prompt
    
    Args:
        numbers: List of available numbers
        target: Target number to reach
        tokenizer: Tokenizer for chat template
    
    Returns:
        Dictionary with prompt and metadata
    """
    # Simple prompt 
    messages = [
        {
            "role": "system",
            "content": "Respond in the following format: <think> ... </think> <answer> ... </answer>"
        },
        {
            "role": "user",
            "content": f"Create an equation using only the numbers {numbers} that equals {target}. "
                       f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, * or /) and each number should only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags"
        }
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Calculate token count
    tokens = tokenizer.encode(prompt)
    token_count = len(tokens)
    
    return {
        "prompt": prompt,
        "target": target,
        "nums": numbers,
        "prompt_tokens": token_count
    }


def prepare_dataset(
    dataset_name: str,
    num_samples: int,
    test_size: float,
    seed: int,
    output_dir: str,
    model_name: str
):
    """
    Prepare and save Countdown dataset with R1 prompts
    
    Args:
        dataset_name: HuggingFace dataset identifier
        num_samples: Number of samples to use
        test_size: Fraction for test set
        seed: Random seed for reproducibility
        output_dir: Directory to save processed dataset
        model_name: Model name for tokenizer
    """
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    
    print(f"Original dataset size: {len(dataset)}")
    
    # Shuffle and select samples
    dataset = dataset.shuffle(seed=seed).select(range(min(num_samples, len(dataset))))
    print(f"Selected {len(dataset)} samples")
    
    # Convert to R1 format
    print("Converting to R1 format...")
    processed_data = []
    for example in tqdm(dataset, desc="Processing"):
        r1_sample = generate_r1_prompt(
            numbers=example["nums"],
            target=example["target"],
            tokenizer=tokenizer
        )
        processed_data.append(r1_sample)
    
    # Split into train and test
    split_idx = int(len(processed_data) * (1 - test_size))
    train_data = processed_data[:split_idx]
    test_data = processed_data[split_idx:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Save to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_file = output_path / "train_countdown_r1.json"
    test_file = output_path / "test_countdown_r1.json"
    
    print(f"Saving train data to {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saving test data to {test_file}...")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # Save metadata
    metadata = {
        "dataset_name": dataset_name,
        "num_samples": num_samples,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "test_size": test_size,
        "seed": seed,
        "model_name": model_name,
    }
    
    metadata_file = output_path / "dataset_metadata.json"
    print(f"Saving metadata to {metadata_file}...")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    # Print example
    print("\n" + "="*80)
    print("EXAMPLE SAMPLE:")
    print("="*80)
    example = train_data[0]
    print(f"Target: {example['target']}")
    print(f"Numbers: {example['nums']}")
    print(f"Prompt tokens: {example['prompt_tokens']}")
    print(f"\nPrompt:\n{example['prompt']}")
    print("="*80)
    
    # Calculate average token count
    avg_tokens = sum(d['prompt_tokens'] for d in train_data) / len(train_data)
    max_tokens = max(d['prompt_tokens'] for d in train_data)
    min_tokens = min(d['prompt_tokens'] for d in train_data)
    
    print(f"\n📊 TOKEN STATISTICS:")
    print(f"  Average: {avg_tokens:.1f} tokens")
    print(f"  Min: {min_tokens} tokens")
    print(f"  Max: {max_tokens} tokens")
    print(f"  max_prompt_length setting: Check training_config.yaml")
    
    print(f"\n✓ Dataset preparation complete!")
    print(f"  Train: {train_file}")
    print(f"  Test: {test_file}")
    print(f"  Metadata: {metadata_file}")


def load_config(config_path: str = "configs/training_config.yaml") -> dict:
    """Load training configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Prepare Countdown dataset for Mini-R1")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config file (default: configs/training_config.yaml)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="HuggingFace dataset name (overrides config)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use (overrides config)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=None,
        help="Fraction for test set (overrides config)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name for tokenizer (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load config file
    config = load_config(args.config)
    
    # Get model name from config (use as-is for tokenizer)
    model_name = config['model']['name']
    
    # Command line args override config values
    prepare_dataset(
        dataset_name=args.dataset_name or config['dataset']['name'],
        num_samples=args.num_samples or config['dataset']['num_samples'],
        test_size=args.test_size or config['dataset']['test_size'],
        seed=args.seed or config['dataset']['shuffle_seed'],
        output_dir=args.output_dir or config['dataset']['cache_dir'],
        model_name=args.model_name or model_name
    )


if __name__ == "__main__":
    main()
