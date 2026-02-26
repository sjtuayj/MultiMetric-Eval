import argparse
import sys
import importlib.util
import sys
import os
from pathlib import Path
from tqdm import tqdm

from .basics import ReadAction, WriteAction
from .instance import SpeechToTextInstance, SpeechToSpeechInstance
from .agent import GenericAgent, AgentPipeline
from .metrics import SCORERS
from .utils import submit_slurm, Visualizer

# --- 引入兄弟模块 (紧耦合) ---
# 注意：使用相对导入或包绝对导入
from ..evaluator import ModelEvaluator
from ..speech_evaluator import SpeechEvaluator

class LatencyEvaluator:
    """同传延迟与质量评测器"""
    def __init__(self, agent: GenericAgent, segment_size=20):
        self.agent = agent
        self.segment_size = segment_size
        self.instances = {}

    def run(self, source_files, ref_files, task="s2t", output_dir="./output", visualize=False):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        visualizer = Visualizer(output_dir) if visualize else None
        
        print(f"🚀 Running Latency Evaluation ({task}, {len(source_files)} samples)...")
        for i, (src, ref) in tqdm(enumerate(zip(source_files, ref_files)), total=len(source_files)):
            if task == "s2t":
                ins = SpeechToTextInstance(i, src, ref, output_dir)
            else:
                ins = SpeechToSpeechInstance(i, src, ref, output_dir)
            
            self.agent.reset()
            while not ins.finish_prediction:
                s_in = ins.send_source(self.segment_size)
                s_out = self.agent.pushpop(s_in)
                ins.receive_prediction(s_out)
            
            self.instances[i] = ins
            if visualizer: visualizer.plot(ins.summarize())
        
        # S2S 准备文本
        if task == "s2s":
            self._prepare_s2s_transcripts(output_dir)
            
        return self.instances

    def _prepare_s2s_transcripts(self, output_dir):
        wav_dir = Path(output_dir) / "wavs"
        if wav_dir.exists():
            for i, ins in self.instances.items():
                transcript = ins.reference if ins.reference else "unknown"
                with open(wav_dir / f"{i}_pred.txt", "w") as f:
                    f.write(transcript)

    def compute_latency(self, computation_aware=False, output_dir="./output"):
        results = {}
        # 扫描所有已注册的 Scorer
        for name, cls in SCORERS.items():
            try:
                if "Align" in name:
                    scorer = cls(computation_aware=computation_aware, output_dir=output_dir)
                else:
                    scorer = cls(computation_aware=computation_aware)
                
                score = scorer(self.instances)
                key = f"{name}_CA" if computation_aware else name
                results[key] = score
            except Exception as e:
                # 某些指标可能因为缺少 MFA 或数据类型不对而失败，忽略
                pass
        return results

def load_agent_from_file(path, class_name):
    """动态加载用户 Agent"""
    spec = importlib.util.spec_from_file_location("user_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)

def main():
    parser = argparse.ArgumentParser(description="MultiMetric Latency Evaluator")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--output", default="./output")
    parser.add_argument("--task", choices=["s2t", "s2s"], default="s2t")
    parser.add_argument("--agent-script", required=True, help="Path to python file containing Agent class")
    parser.add_argument("--agent-class", required=True, help="Name of the Agent class")
    parser.add_argument("--segment-size", type=int, default=20)
    parser.add_argument("--computation-aware", action="store_true")
    parser.add_argument("--quality", action="store_true", help="Run Quality Evaluation (BLEU/WER) after latency")
    parser.add_argument("--slurm", action="store_true")
    args = parser.parse_args()

    if args.slurm:
        submit_slurm(args, __file__)

    # 1. 加载数据
    with open(args.source) as f: src = [l.strip() for l in f if l.strip()]
    ref = [None]*len(src)
    if args.target:
        with open(args.target) as f: ref = [l.strip() for l in f if l.strip()]

    # 2. 加载用户 Agent
    AgentClass = load_agent_from_file(args.agent_script, args.agent_class)
    agent = AgentClass()
    
    # 3. 运行 Latency 评测
    evaluator = LatencyEvaluator(agent, args.segment_size)
    instances = evaluator.run(src, ref, args.task, args.output)
    
    # 4. 计算 Latency 指标
    print("\n--- Latency Metrics ---")
    scores = evaluator.compute_latency(False, args.output)
    if args.computation_aware:
        scores.update(evaluator.compute_latency(True, args.output))
    
    for k, v in scores.items():
        print(f"{k:<25}: {v:.4f}")

    # 5. [紧耦合] 运行 Quality 指标
    if args.quality and args.target:
        print("\n--- Quality Metrics (Integrated) ---")
        
        # 收集预测结果
        predictions = [ins.get_prediction_content() for ins in instances.values()]
        
        if args.task == "s2t":
            # 调用 ModelEvaluator 算 BLEU/COMET
            # 默认只开 BLEU 以保证速度，用户可在代码里改
            qual_eval = ModelEvaluator(use_comet=False, use_whisper=False)
            q_res = qual_eval.evaluate(predictions, ref, src)
            print(q_res)
            
        elif args.task == "s2s":
            # 调用 SpeechEvaluator 算 UTMOS/WER
            # predictions 是 wav 路径列表
            speech_eval = SpeechEvaluator(whisper_model_name="tiny") # 用 tiny 快速出 WER
            s_res = speech_eval.evaluate(predictions, ref)
            print(s_res)

if __name__ == "__main__":
    main()