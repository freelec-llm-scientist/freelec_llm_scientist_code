"""
완전한 생성 테스트 - 모든 설정 포함
모델이 제대로 작동하는지 학습 전에 검증
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random


def test_generation_complete():
    """완전한 생성 테스트"""
    
    print("="*80)
    print("🔧 LOADING MODEL...")
    print("="*80)
    
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    # 1. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # 2. pad_token 설정 (중요!)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Tokenizer loaded")
    print(f"  - Vocab size: {len(tokenizer)}")
    print(f"  - Pad token: {tokenizer.pad_token}")
    print(f"  - EOS token: {tokenizer.eos_token}")
    
    # 3. 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # bf16 사용
        device_map="auto"
    )
    
    print(f"✓ Model loaded")
    print(f"  - Dtype: {model.dtype}")
    print(f"  - Device: {model.device}")
    print(f"  - Vocab size: {model.config.vocab_size}")
    
    # 4. 검증 및 리사이즈
    if len(tokenizer) != model.config.vocab_size:
        print(f"⚠️  Vocab size mismatch!")
        print(f"   Tokenizer: {len(tokenizer)}")
        print(f"   Model: {model.config.vocab_size}")
        print(f"   → Resizing model embeddings...")
        model.resize_token_embeddings(len(tokenizer))
        print(f"   ✓ Resized to {len(tokenizer)}")
    
    print("\n" + "="*80)
    print("📝 CREATING PROMPT...")
    print("="*80)
    
    # 5. 테스트 문제 (학습 데이터와 동일한 형식)
    numbers = [75, 25, 3, 1, 7, 10]
    target = 111
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first think about the reasoning process in <think></think> tags and then provide the answer in <answer></answer> tags."
        },
        {
            "role": "user",
            "content": f"Using the numbers {numbers}, create an equation that equals {target}. "
                      f"You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
                      f"Think step by step in <think> tags, then provide your final equation in <answer> tags."
        }
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(prompt)
    print("\n" + "="*80)
    print("🚀 GENERATING (3 attempts with different settings)...")
    print("="*80)
    
    # 6. 입력 준비
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    
    # 7. 세 가지 설정으로 테스트
    test_configs = [
        {
            "name": "Safe (repetition_penalty=1.2)",
            "params": {
                "max_new_tokens": 300,
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 50,
                "do_sample": True,
                "repetition_penalty": 1.2,  # 🔥 반복 방지
                "no_repeat_ngram_size": 3,  # 3-gram 반복 방지
            }
        },
        {
            "name": "Moderate (repetition_penalty=1.1)",
            "params": {
                "max_new_tokens": 300,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 2,
            }
        },
        {
            "name": "Original (학습 코드 설정)",
            "params": {
                "max_new_tokens": 300,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True,
                # repetition_penalty 없음!
            }
        }
    ]
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/3: {config['name']}")
        print(f"{'='*80}")
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **config['params'],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 디코딩
        completion = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # 출력
        print(f"\n📄 OUTPUT ({len(completion)} chars):")
        print("-"*80)
        print(completion[:500])  # 처음 500자만
        if len(completion) > 500:
            print(f"... (truncated, total {len(completion)} chars)")
        print("-"*80)
        
        # 검증
        has_think = "<think>" in completion and "</think>" in completion
        has_answer = "<answer>" in completion and "</answer>" in completion
        has_numbers = any(str(n) in completion for n in numbers)
        has_operators = any(op in completion for op in ['+', '-', '*', '/'])
        is_repetitive = any(c*20 in completion for c in set(completion[:100]))
        
        print(f"\n✓ Validation:")
        print(f"  - Has <think> tags: {'✅' if has_think else '❌'}")
        print(f"  - Has <answer> tags: {'✅' if has_answer else '❌'}")
        print(f"  - Contains problem numbers: {'✅' if has_numbers else '❌'}")
        print(f"  - Contains operators: {'✅' if has_operators else '❌'}")
        print(f"  - NOT repetitive: {'✅' if not is_repetitive else '❌ REPETITIVE!'}")
        
        # 점수
        score = sum([has_think, has_answer, has_numbers, has_operators, not is_repetitive])
        print(f"\n🎯 Score: {score}/5")
        
        if score >= 4:
            print("✅ GOOD - This configuration works!")
        elif score >= 2:
            print("⚠️  OKAY - Partially working")
        else:
            print("❌ BAD - Not working properly")


if __name__ == "__main__":
    test_generation_complete()
