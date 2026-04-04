"""Analyze latest TensorBoard event file for key Mini-R1 training metrics.

Usage:
  uv run python scripts/analyze_tensorboard.py 

Outputs a tabular summary + quick qualitative assessment.
"""
from pathlib import Path
import os
from typing import Dict, List, Tuple

from tensorboard.backend.event_processing import event_accumulator

# Default log directory (adjust if needed)
LOG_DIR = "logs/miniR1_251120_generation_samples/tensorboard_qwen2.5-3B_25112"

METRIC_KEYS: List[str] = [
    "train/loss",
    "train/learning_rate",
    "train/rewards/format_reward_func",
    "train/rewards/equation_reward_func",
    "train/rewards/mean",
    "train/objective/kl",
    "train/objective/entropy",
    "train/policy/approx_kl",
    "train/policy/clipfrac",
    "train/completions/max_length",
]

def pick_latest_event_file(log_dir: str) -> Path:
    files = [f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents.")]
    if not files:
        raise FileNotFoundError(f"No event files found in {log_dir}")
    # 파일명 형태: events.out.tfevents.<timestamp>.<hostname>.<pid>.<rand>
    # timestamp는 세번째 토큰이 아니라 네번째(인덱스=3)일 수 있으므로 숫자 추출을 유연하게 처리
    parsed = []
    for fname in files:
        parts = fname.split('.')
        ts = None
        for p in parts:
            if p.isdigit():
                ts = int(p)
                break
        if ts is None:
            continue
        parsed.append((ts, fname))
    if not parsed:
        raise RuntimeError("Could not parse timestamps from event files.")
    parsed.sort(key=lambda x: x[0])
    return Path(log_dir) / parsed[-1][1]


def compute_interval_stats(steps: List[int], vals: List[float], intervals: List[Tuple[int, int]]) -> Dict:
    """구간별 평균 계산"""
    interval_avgs = []
    for (start, end) in intervals:
        interval_vals = [v for s, v in zip(steps, vals) if start <= s <= end]
        if interval_vals:
            interval_avgs.append(sum(interval_vals) / len(interval_vals))
        else:
            interval_avgs.append(None)
    return interval_avgs


def load_scalars(event_file: Path) -> Dict[str, Dict[str, float]]:
    ea = event_accumulator.EventAccumulator(str(event_file), size_guidance={
        event_accumulator.SCALARS: 200000,
    })
    ea.Reload()
    available_list = ea.Tags().get("scalars", [])
    available = set(available_list)
    
    # 5개 구간 정의: [1-200], [201-400], [401-600], [601-800], [801-1035]
    intervals = [(1, 200), (201, 400), (401, 600), (601, 800), (801, 1035)]
    
    data = {}
    for key in METRIC_KEYS:
        if key in available:
            scalars = ea.Scalars(key)
            if not scalars:
                continue
            steps = [s.step for s in scalars]
            vals = [s.value for s in scalars]
            data[key] = {
                "last_step": steps[-1],
                "last_value": vals[-1],
                "min": min(vals),
                "max": max(vals),
                "avg": sum(vals) / len(vals),
                "count": len(vals),
                "intervals": compute_interval_stats(steps, vals, intervals),
            }
    # 추가: 모든 reward 관련 지표 자동 수집 (키워드: 'reward')
    reward_keys = [k for k in available_list if 'reward' in k]
    for rk in reward_keys:
        if rk in data:  # 이미 기본 목록에 포함된 경우 skip
            continue
        scalars = ea.Scalars(rk)
        if not scalars:
            continue
        steps = [s.step for s in scalars]
        vals = [s.value for s in scalars]
        data[rk] = {
            "last_step": steps[-1],
            "last_value": vals[-1],
            "min": min(vals),
            "max": max(vals),
            "avg": sum(vals) / len(vals),
            "count": len(vals),
            "intervals": compute_interval_stats(steps, vals, intervals),
        }
    # reward 키 목록을 함께 반환할 수 있도록 special entry 저장(요약용)
    data['__reward_keys__'] = { 'keys': reward_keys }
    data['__intervals__'] = { 'ranges': intervals }
    return data


def summarize(data: Dict[str, Dict[str, float]]):
    intervals_meta = data.get('__intervals__', {})
    intervals = intervals_meta.get('ranges', [])
    
    print("=== TensorBoard Metrics Summary (Latest File) ===")
    print("\n구간별 평균 (5개 구간):")
    for i, (start, end) in enumerate(intervals, 1):
        print(f"  구간{i}: step {start:4d}-{end:4d}")
    print()
    
    header = f"{'metric':40s} {'step':>6s} {'last':>10s} {'avg':>10s} {'구간1':>10s} {'구간2':>10s} {'구간3':>10s} {'구간4':>10s} {'구간5':>10s}"
    print(header)
    print("-" * len(header))
    printed = set()
    for k in METRIC_KEYS:
        v = data.get(k)
        if not v:
            continue
        ivs = v.get('intervals', [None]*5)
        iv_str = "".join([f"{x:10.4f}" if x is not None else "      N/A " for x in ivs])
        print(f"{k:40s} {v['last_step']:6d} {v['last_value']:10.4f} {v['avg']:10.4f} {iv_str}")
        printed.add(k)

    # reward 관련 추가 키 출력
    reward_meta = data.get('__reward_keys__', {})
    extra_rewards = [rk for rk in reward_meta.get('keys', []) if rk not in printed and rk in data]
    if extra_rewards:
        print("\n-- Additional Reward Scalars --")
        for rk in sorted(extra_rewards):
            v = data[rk]
            ivs = v.get('intervals', [None]*5)
            iv_str = "".join([f"{x:10.4f}" if x is not None else "      N/A " for x in ivs])
            print(f"{rk:40s} {v['last_step']:6d} {v['last_value']:10.4f} {v['avg']:10.4f} {iv_str}")

    # Qualitative assessment
    print("\n=== Quick Assessment ===")
    loss = data.get("train/loss")
    eq = data.get("train/rewards/equation_reward_func/mean")
    fmt = data.get("train/rewards/format_reward_func/mean")
    maxlen = data.get("train/completions/max_length")
    kl = data.get("train/objective/kl")
    entropy = data.get("train/objective/entropy")

    if loss:
        if loss["last_value"] > loss["avg"] * 1.15:
            print("- Loss: 최근 값이 평균보다 높음 → 과적합 감소 또는 학습률 재조정 필요 가능")
        elif loss["last_value"] < loss["avg"] * 0.85:
            print("- Loss: 최근 값이 평균보다 훨씬 낮음 → 급격한 수렴; 검증 필요")
        else:
            print("- Loss: 평균 근처 → 안정적")

    if fmt:
        if fmt["last_value"] >= 0.45:
            print("- Format reward: 태그/형식 수렴")
        else:
            print("- Format reward: 아직 일부 샘플 불안정")

    if eq:
        ivs = eq.get('intervals', [])
        if len(ivs) >= 5 and ivs[0] is not None and ivs[-1] is not None:
            trend = "상승" if ivs[-1] > ivs[0] * 1.2 else ("하락" if ivs[-1] < ivs[0] * 0.8 else "유지")
            print(f"- Equation reward: 구간1 {ivs[0]:.3f} → 구간5 {ivs[-1]:.3f} ({trend})")

    if maxlen:
        saturation_ratio = maxlen["last_value"] / maxlen["max"] if maxlen["max"] else 0
        if saturation_ratio > 0.95:
            print("- Max length: 상한 근접 반복 → max_completion_length 확장 고려")
        else:
            print("- Max length: 아직 여유 있음")

    if kl and kl["last_value"] > kl["avg"] * 1.3:
        print("- KL: 최근 급상승 → 정책 변동 과다, beta 증가 검토")
    if entropy and entropy["last_value"] < entropy["avg"] * 0.7:
        print("- Entropy: 최근 크게 감소 → 탐색 축소, 온도 또는 top_p 재조정 고려")
    print("===============================================")


def main():
    event_file = pick_latest_event_file(LOG_DIR)
    print(f"Using latest event file: {event_file}")
    data = load_scalars(event_file)
    if not data:
        print("No target metrics found.")
        return
    summarize(data)

if __name__ == "__main__":
    main()
