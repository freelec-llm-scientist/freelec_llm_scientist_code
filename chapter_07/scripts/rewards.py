"""
Reward Functions for Mini-R1 GRPO Training
"""

import re
from typing import List
import ast
import operator
import logging

logger = logging.getLogger(__name__)


def format_reward_func(
    prompts: List[str],
    completions: List[str],
    completion_ids: List[List[int]],
    **kwargs
) -> List[float]:
    """
    형식 점수
    - <think>...</think> <answer>...</answer> 구조 확인
    - 최대: 0.5점
    """
    rewards = []
    
    for completion in completions:
        try:
            score = 0.0
            
            # 필수 태그 체크
            has_think_start = "<think>" in completion
            has_think_end = "</think>" in completion
            has_answer_start = "<answer>" in completion
            has_answer_end = "</answer>" in completion
            
            if not (has_think_start and has_think_end and has_answer_start and has_answer_end):
                rewards.append(0.0)
                continue
            
            # 순서 검증
            think_start = completion.find("<think>")
            think_end = completion.find("</think>")
            answer_start = completion.find("<answer>")
            answer_end = completion.find("</answer>")
            
            if think_start < think_end < answer_start < answer_end:
                score = 0.5
            else:
                score = 0.1
            
            rewards.append(score)
                
        except Exception as e:
            logger.warning(f"Format reward error: {e}")
            rewards.append(0.0)
    
    return rewards


def safe_eval(expr: str) -> float:
    """
    안전한 수식 계산
    허용된 연산자: +, -, *, /
    """
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
    }
    
    def eval_node(node):
        if isinstance(node, ast.Constant):
            return float(node.value)
        elif isinstance(node, ast.Num):
            return float(node.n)
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"Operator not allowed: {op_type}")
            left = eval_node(node.left)
            right = eval_node(node.right)
            return allowed_operators[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"Unary operator not allowed: {op_type}")
            operand = eval_node(node.operand)
            return allowed_operators[op_type](operand)
        else:
            raise ValueError(f"Node type not allowed: {type(node)}")
    
    try:
        tree = ast.parse(expr, mode='eval')
        return eval_node(tree.body)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


def equation_reward_func(
    prompts: List[str],
    completions: List[str],
    completion_ids: List[List[int]],
    **kwargs
) -> List[float]:
    """
    수식 정확도 리워드
    - 정답: 2.0점
    - 숫자는 맞게 사용했지만 오답: 0.1점
    - 그 외: 0.0점
    """
    target = kwargs.get('target', [])
    nums = kwargs.get('nums', [])
    
    if not target or not nums:
        logger.warning("Missing target or nums in equation_reward_func")
        return [0.0] * len(completions)
    
    rewards = []
    
    for completion, gt, numbers in zip(completions, target, nums):
        try:
            # <answer> 태그 내용 추출
            match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            if match is None:
                rewards.append(0.0)
                continue
            
            equation = match.group(1).strip()
            
            if not equation:
                rewards.append(0.0)
                continue
            
            # = 기호 있으면 왼쪽만 추출
            if '=' in equation:
                equation = equation.split('=')[0].strip()
            
            # 허용된 문자만 있는지 검증
            allowed_pattern = r'^[\d+\-*/().\s]+$'
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)
                continue

            # 사용된 숫자 검증
            tokens = re.findall(r'\d+', equation)
            used_numbers = sorted([int(t) for t in tokens])
            expected_numbers = sorted(numbers)
            
            if used_numbers != expected_numbers:
                rewards.append(0.1)  # 숫자를 잘못 사용
                continue
            
            # 수식 계산
            try:
                result = safe_eval(equation)
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(2.0)  # 정답!
                else:
                    rewards.append(0.1)  # 숫자는 맞지만 계산 오류
            except (ValueError, ZeroDivisionError, SyntaxError):
                rewards.append(0.0)
                
        except Exception as e:
            logger.warning(f"Equation reward error: {e}")
            rewards.append(0.0)
    
    return rewards


def length_penalty_func(
    prompts: List[str],
    completions: List[str],
    completion_ids: List[List[int]],
    **kwargs
) -> List[float]:
    """
    길이 페널티
    - 1536 토큰 초과 시: -0.5점
    - 그 외: 0.0점
    """
    max_completion_length = kwargs.get('max_completion_length', 1536)
    
    rewards = []
    safe_buffer = 100
    
    for completion in completions:
        length = len(completion)
        
        if length < (max_completion_length - safe_buffer):
            rewards.append(0.0)
        else:
            rewards.append(-0.5)
    
    return rewards