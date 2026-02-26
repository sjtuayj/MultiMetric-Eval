"""
test_real_latency.py
使用真实数据集 (zh-en-littleprince) 验证 Latency 模块功能 (S2S 模式)
"""

import os
import shutil
import glob
from pathlib import Path

# 导入你的库
from multimetric_eval import load_dataset
from multimetric_eval.latency import GenericAgent, ReadAction, WriteAction, LatencyEvaluator
from multimetric_eval.latency.metrics import SCORERS 

# ================= 1. 配置 =================
DATASET_NAME = "zh-en-littleprince"
AUDIO_DIR = Path("./datasets/zh-en-littleprince/audio")
OUTPUT_DIR = Path("./latency_real_output")
NUM_SAMPLES = 27  # 只测前 27 条 (英文部分)

# ================= 2. 定义 Agent =================
class EchoAgent(GenericAgent):
    """
    Echo Agent: 
    完全读完音频后，把同样的音频“复读”出来 (模拟 S2S)。
    这是一种最简单的策略，延迟会很大 (StartOffset ≈ Source Duration)。
    """
    def policy(self, states=None):
        if not self.states.source_finished:
            return ReadAction()
        
        if not self.states.target_finished:
            # 这里的 source 是一个 float list (采样点)
            # 我们直接把读到的所有音频作为输出返回
            return WriteAction(self.states.source, finished=True)
            
        return ReadAction()

# ================= 3. 数据准备 =================
print("🚀 准备数据...")

if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir()

# 加载数据集 (获取 Reference Text)
dataset = load_dataset(DATASET_NAME)
# 小王子数据集前 27 条通常是英文 (具体取决于你的数据集排序，这里假设前 27 条对应音频)
ref_texts = dataset.reference_texts[:NUM_SAMPLES]

# 加载音频文件
all_wavs = sorted(glob.glob(str(AUDIO_DIR / "*.wav")))
source_wavs = all_wavs[:NUM_SAMPLES]

print(f"   音频文件数: {len(source_wavs)}")
print(f"   参考文本数: {len(ref_texts)}")
assert len(source_wavs) == len(ref_texts), "数据数量不匹配"

# ================= 4. 运行评测 =================
print("\n🏃‍♂️ 开始运行 Latency 评测 (Simulation)...")

agent = EchoAgent()
evaluator = LatencyEvaluator(agent, segment_size=40) # 40ms 步长

instances = evaluator.run(
    source_files=source_wavs,
    ref_files=ref_texts,
    task="s2s",
    output_dir=str(OUTPUT_DIR),
    visualize=False 
)

print("\n💾 强制写入音频文件 (确保 MFA 有数据)...")
# 这里的 .wav 是 Agent 输出的音频 (即复读的音频)
for i, ins in instances.items():
    path = ins.get_prediction_content()
    # print(f"   Sample {i}: {path}") 

print("\n📊 计算指标...")
try:
    # 尝试计算所有指标 (包含 MFA 对齐)
    # output_dir 必须传，MFA 要用
    scores = evaluator.compute_latency(computation_aware=True, output_dir=str(OUTPUT_DIR))
    
    print("-" * 40)
    for k, v in scores.items():
        print(f"{k:<30}: {v:.4f}")
    print("-" * 40)

    # 验证 CustomATD
    if "CustomATD_CA" in scores:
        print("✅ CustomATD_CA 计算成功。")
    else:
        print("❌ CustomATD_CA 缺失 (可能计算出错)。")

    # 验证 MFA
    if "StartOffset_SpeechAlign_CA" in scores:
        print("✅ MFA StartOffset 计算成功 (说明离线环境配置正确)。")
    else:
        print("⚠️ MFA 指标缺失 (说明 'mfa align' 命令失败或未安装)。")

except Exception as e:
    print(f"❌ 计算过程报错: {e}")
    import traceback
    traceback.print_exc()

print(f"\n✅ 测试结束。结果目录: {OUTPUT_DIR}")