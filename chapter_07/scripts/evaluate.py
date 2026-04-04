"""
Model Evaluation Script for Mini-R1
체크포인트 테스트 및 샘플 생성
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

from rewards import format_reward_func, equation_reward_func


# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(checkpoint_path: str, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """
    체크포인트에서 모델과 토크나이저 로드
    
    Args:
        checkpoint_path: 체크포인트 디렉토리 경로
        base_model: 베이스 모델 이름
    
    Returns:
        model, tokenizer
    """
    logger.info(f"Loading tokenizer from {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    logger.info(f"Loading base model from {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    logger.info(f"Loading LoRA weights from {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    
    return model, tokenizer


def generate_sample(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    단일 샘플 생성
    
    Args:
        model: 모델
        tokenizer: 토크나이저
        prompt: 입력 프롬프트
        max_new_tokens: 최대 생성 토큰 수
        temperature: 샘플링 온도
        top_p: nucleus sampling p값
    
    Returns:
        생성된 텍스트
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # 프롬프트 부분 제거하고 생성된 부분만 반환
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text


def evaluate_checkpoint(
    checkpoint_path: str,
    test_data_path: str = ".cache/datasets/test_countdown_r1.json",
    num_samples: int = 100,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_file: str = None,
) -> dict:
    """
    체크포인트 평가
    
    Args:
        checkpoint_path: 체크포인트 경로
        test_data_path: 테스트 데이터 JSON 파일
        num_samples: 평가할 샘플 수
        base_model: 베이스 모델 이름
        output_file: 결과 저장 파일 (없으면 자동 생성)
    
    Returns:
        평가 메트릭 딕셔너리
    """
    # 모델 로드
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, base_model)
    
    # 테스트 데이터 로드
    logger.info(f"Loading test data from {test_data_path}")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 샘플 수 제한
    test_data = test_data[:num_samples]
    logger.info(f"Evaluating on {len(test_data)} samples")
    
    # 평가 실행
    results = []
    completions = []
    targets = []
    nums_list = []
    
    for example in tqdm(test_data, desc="Generating"):
        prompt = example['prompt']
        target = example['target']
        nums = example['nums']
        
        # 생성
        completion = generate_sample(model, tokenizer, prompt)
        
        completions.append(completion)
        targets.append(str(target))
        nums_list.append(nums)
        
        results.append({
            'prompt': prompt,
            'completion': completion,
            'target': target,
            'nums': nums,
        })
    
    # 보상 계산
    logger.info("Computing rewards...")
    format_rewards = format_reward_func(completions, targets)
    equation_rewards = equation_reward_func(completions, targets, nums_list)
    combined_rewards = [f + e for f, e in zip(format_rewards, equation_rewards)]
    
    # 메트릭 계산
    format_accuracy = sum(format_rewards) / len(format_rewards)
    equation_accuracy = sum(equation_rewards) / len(equation_rewards)
    success_rate = sum(1 for r in combined_rewards if r > 1.5) / len(combined_rewards)
    avg_reward = sum(combined_rewards) / len(combined_rewards)
    
    metrics = {
        'checkpoint': checkpoint_path,
        'num_samples': len(test_data),
        'format_accuracy': format_accuracy,
        'equation_accuracy': equation_accuracy,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
    }
    
    logger.info("=" * 80)
    logger.info("Evaluation Results:")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Samples: {len(test_data)}")
    logger.info(f"  Format Accuracy: {format_accuracy:.2%}")
    logger.info(f"  Equation Accuracy: {equation_accuracy:.2%}")
    logger.info(f"  Success Rate: {success_rate:.2%}")
    logger.info(f"  Average Reward: {avg_reward:.2f}")
    logger.info("=" * 80)
    
    # 결과 저장
    if output_file is None:
        checkpoint_name = Path(checkpoint_path).name
        output_file = f"evaluation_{checkpoint_name}.json"
    
    output_data = {
        'metrics': metrics,
        'samples': results[:10],  # 처음 10개 샘플만 저장
    }
    
    output_path = Path("logs") / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    # 성공 샘플 예시 출력
    logger.info("\n" + "=" * 80)
    logger.info("Example Successful Samples:")
    logger.info("=" * 80)
    
    success_count = 0
    for i, (result, reward) in enumerate(zip(results, combined_rewards)):
        if reward > 1.5 and success_count < 3:
            logger.info(f"\n--- Sample {i+1} (Reward: {reward:.2f}) ---")
            logger.info(f"Target: {result['target']}")
            logger.info(f"Numbers: {result['nums']}")
            logger.info(f"Completion:\n{result['completion']}")
            logger.info("-" * 80)
            success_count += 1
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mini-R1 checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=".cache/datasets/test_countdown_r1.json",
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file name (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        test_data_path=args.test_data,
        num_samples=args.num_samples,
        base_model=args.base_model,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
