"""
Mini-R1 GRPO Training Script (Weighted Rewards Version)
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
import yaml
import functools

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

# reward.py에서 함수 import
from rewards import format_reward_func, equation_reward_func, length_penalty_func

# 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Helper: 가중치 적용 래퍼 함수
# ------------------------------------------------------------------------------
def get_weighted_reward_func(reward_func, weight, **fixed_kwargs):
    """
    기존 리워드 함수의 결과값에 가중치(weight)를 곱해서 반환하는 함수 생성
    fixed_kwargs: 함수에 고정으로 전달할 추가 파라미터 (예: max_completion_length)
    """
    @functools.wraps(reward_func)
    def wrapper(*args, **kwargs):
        # fixed_kwargs를 먼저 적용하고, 호출 시 전달된 kwargs로 덮어씀
        merged_kwargs = {**fixed_kwargs, **kwargs}
        rewards = reward_func(*args, **merged_kwargs)
        return [r * weight for r in rewards]
    return wrapper

# ------------------------------------------------------------------------------
# Main Logic
# ------------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_dataset_from_json(file_path: str) -> Dataset:
    with open(file_path, 'r', encoding='utf-8') as f:
        return Dataset.from_list(json.load(f))

def save_training_metrics(metrics: dict, output_file: str):
    output_path = Path(output_file)
    if output_path.exists():
        with open(output_path, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = []
    all_metrics.append({"timestamp": datetime.now().isoformat(), **metrics})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)


class MemoryClearCallback(TrainerCallback):
    """
    주기적으로 CUDA 캐시 정리
    """
    def __init__(self, clear_every_n_steps: int = 10):
        self.clear_every_n_steps = clear_every_n_steps

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.clear_every_n_steps == 0:
            torch.cuda.empty_cache()
            logger.debug(f"🧹 Cleared CUDA cache at step {state.global_step}")
        return control


class SampleSavingCallback(TrainerCallback):
    """
    학습 중 생성 샘플 저장 콜백 (가중치 적용된 점수 계산 포함)
    """
    def __init__(self, save_steps: int, tokenizer, config, eval_dataset, weights, max_completion_length: int):
        self.save_steps = save_steps
        self.tokenizer = tokenizer
        self.config = config
        self.eval_dataset = eval_dataset
        self.weights = weights
        self.max_completion_length = max_completion_length

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        step = state.global_step
        if step == 1 or (step % self.save_steps == 0):
            model = kwargs.get('model', None)
            if model and self.tokenizer:
                self._generate_and_save_samples(model, step)
        return control

    def _generate_and_save_samples(self, model, step):
        import random
        try:
            was_training = model.training
            model.eval()
    
            # 샘플 데이터 준비
            if self.eval_dataset and len(self.eval_dataset) > 0:
                sample = self.eval_dataset[random.randint(0, len(self.eval_dataset)-1)]
                numbers = sample.get('nums', []) or sample.get('numbers', [])
                target = sample.get('target', None)
            else:
                numbers = random.sample(range(1, 101), 6)
                target = random.randint(10, 999)
    
            # 프롬프트 생성
            messages = [
                {"role": "system", "content": "Respond in the following format: <think> ... </think> <answer> ... </answer>"},
                {"role": "user", "content":  f"Create an equation using only the numbers {numbers} that equals {target}. "
                       f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, * or /) and each number should only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags"}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
            # 생성
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=self.max_completion_length,
                    temperature=self.config['grpo'].get('temperature', 0.9),
                    top_p=self.config['grpo'].get('top_p', 0.95),
                    do_sample=True
                )
            
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            completion = full_output[len(prompt):]
    
            # === [수정] GRPOTrainer가 호출하는 것과 동일한 방식으로 호출 ===
            # prompts, completions, completion_ids를 모두 리스트로 전달
            prompts = [prompt]
            completions = [completion]
            prompt_length = inputs['input_ids'].shape[1]
            completion_ids = [outputs[0, prompt_length:].tolist()]
            
            raw_format = format_reward_func(
                prompts,
                completions,
                completion_ids
            )[0]
            
            raw_equation = equation_reward_func(
                prompts,
                completions,
                completion_ids,
                target=[target],
                nums=[numbers]
            )[0]
            
            raw_length = length_penalty_func(
                prompts,
                completions,
                completion_ids,
                max_completion_length=self.max_completion_length
            )[0]
    
            # 가중치 적용
            w_format = raw_format * self.weights['format']
            w_equation = raw_equation * self.weights['equation']
            w_length = raw_length * self.weights['length']
            
            total_reward = w_format + w_equation + w_length
    
            # 파일 저장
            samples_dir = Path("logs/generation_samples")
            samples_dir.mkdir(parents=True, exist_ok=True)
            sample_file = samples_dir / f"step_{step:05d}.txt"
    
            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write(f"Step: {step}\nTarget: {target}, Nums: {numbers}\n")
                f.write(f"Generated:\n{completion}\n\n")
                f.write(f"--- Rewards (Weighted) ---\n")
                f.write(f"Format:   {w_format:.2f} (Raw: {raw_format:.2f} * {self.weights['format']})\n")
                f.write(f"Equation: {w_equation:.2f} (Raw: {raw_equation:.2f} * {self.weights['equation']})\n")
                f.write(f"Length:   {w_length:.2f} (Raw: {raw_length:.2f} * {self.weights['length']})\n")
                f.write(f"Total:    {total_reward:.2f}\n")
    
            if was_training: model.train()
            logger.info(f"✅ Step {step} Sample Saved. Total Reward: {total_reward:.2f}")
    
        except Exception as e:
            logger.error(f"Sample generation failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    # Config 로드
    config = load_config(args.config)
    set_seed(config.get('dataset', {}).get('shuffle_seed', 42))
    
    # 데이터셋 로드
    dataset_dir = Path(config['dataset']['cache_dir'])
    train_dataset = load_dataset_from_json(str(dataset_dir / "train_countdown_r1.json"))
    test_dataset = load_dataset_from_json(str(dataset_dir / "test_countdown_r1.json"))
    
    # 모델 & 토크나이저 로드
    model_name = config['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=config['model'].get('attn_implementation', 'eager')
    )
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # LoRA 설정
    use_peft = config['model'].get('use_peft', False)  # Config 기본값에 맞춤
    peft_config = None
    if use_peft:
        pc = config.get('peft', {})
        peft_config = LoraConfig(
            r=pc.get('r', 16),
            lora_alpha=pc.get('lora_alpha', 32),
            lora_dropout=pc.get('lora_dropout', 0.05),
            target_modules=pc.get('target_modules', None),
            task_type="CAUSAL_LM",
        )
        logger.info(f"✅ Using LoRA with r={pc.get('r', 16)}")
    else:
        logger.info("⚠️  Full fine-tuning mode (no LoRA)")

    # === [중요] 리워드 가중치 로드 및 함수 래핑 ===
    reward_weights_config = config.get('reward_weights', {})
    grpo_config = config['grpo']
    max_completion_length = grpo_config['max_completion_length']
    
    # Config에서 값 읽기 (없으면 기본값 1.0)
    w_format = reward_weights_config.get('format_reward', 1.0)
    w_equation = reward_weights_config.get('equation_reward', 1.0)
    w_length = reward_weights_config.get('length_penalty', 1.0)
    
    logger.info(f"{'='*40}")
    logger.info(f"⚖️  Reward Weights Applied:")
    logger.info(f"   - Format:   x {w_format}")
    logger.info(f"   - Equation: x {w_equation}")
    logger.info(f"   - Length:   x {w_length}")
    logger.info(f"{'='*40}")

    # === [수정] max_completion_length를 고정 파라미터로 전달 ===
    weighted_format_func = get_weighted_reward_func(format_reward_func, w_format)
    weighted_equation_func = get_weighted_reward_func(equation_reward_func, w_equation)
    weighted_length_func = get_weighted_reward_func(
        length_penalty_func, 
        w_length, 
        max_completion_length=max_completion_length  # 고정 파라미터
    )

    # Trainer 설정
    training_config = config['training']
    
    training_args = GRPOConfig(
        output_dir=training_config['output_dir'],
        learning_rate=training_config['learning_rate'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        warmup_ratio=training_config.get('warmup_ratio', 0.05),
        max_steps=training_config['max_steps'],
        logging_steps=training_config['logging_steps'],
        save_steps=training_config['save_steps'],
        eval_steps=training_config.get('eval_steps', 50),  # === [추가] ===
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        gradient_checkpointing=training_config['gradient_checkpointing'],
        bf16=training_config.get('bf16', True),
        max_prompt_length=grpo_config['max_prompt_length'],
        max_completion_length=max_completion_length,
        num_generations=grpo_config['num_generations'],
        temperature=grpo_config.get('temperature', 0.9),  # === [추가] ===
        top_p=grpo_config.get('top_p', 0.95),  # === [추가] ===
        loss_type=grpo_config['loss_type'],
        scale_rewards=grpo_config['scale_rewards'],
        beta=grpo_config['beta'],
        logging_dir=training_config.get('logging_dir'),
        report_to=training_config.get('report_to', ['tensorboard']),
        save_total_limit=training_config.get('save_total_limit', 2),
        max_grad_norm=training_config.get('max_grad_norm', 0.5),
        optim=training_config.get('optim', 'adamw_torch'),
        weight_decay=training_config.get('weight_decay', 0.01),
    )
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        reward_funcs=[
            weighted_format_func,
            weighted_equation_func,
            weighted_length_func,  # max_completion_length 이미 고정됨
        ],
    )
    
    # === [추가] 콜백들 ===
    weights_dict = {'format': w_format, 'equation': w_equation, 'length': w_length}
    
    # 샘플 저장 콜백
    sample_callback = SampleSavingCallback(
        save_steps=config['sampling']['save_samples_every'],
        tokenizer=tokenizer,
        config=config,
        eval_dataset=test_dataset,
        weights=weights_dict,
        max_completion_length=max_completion_length
    )
    trainer.add_callback(sample_callback)
    
    # 메모리 정리 콜백
    clear_cache_steps = config.get('runpod', {}).get('clear_cache_every_n_steps', 10)
    memory_callback = MemoryClearCallback(clear_every_n_steps=clear_cache_steps)
    trainer.add_callback(memory_callback)
    
    logger.info("🚀 Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("✨ Training completed.")

if __name__ == "__main__":
    main()
