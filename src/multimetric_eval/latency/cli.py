import argparse
import sys
import importlib.util
import sys
import os
from pathlib import Path
from tqdm import tqdm
from .utils import submit_slurm, Visualizer

from .basics import ReadAction, WriteAction
from .instance import SpeechToTextInstance, SpeechToSpeechInstance
from .agent import GenericAgent, AgentPipeline
from .metrics import SCORERS
from .utils import submit_slurm, Visualizer

# --- 引入兄弟模块 (紧耦合) ---
# 注意：使用相对导入或包绝对导入
from ..translation_evaluator import TranslationEvaluator

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
                import time
                t0 = time.perf_counter()
                s_out = self.agent.pushpop(s_in)
                t1 = time.perf_counter()
                ins.add_inference_time(t1 - t0)
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

        # 增加 show_all_metrics 开关，默认为 False
    def compute_latency(self, computation_aware=False, output_dir="./output", show_all_metrics=False):
        results = {}
        # 1. 扫描并算出所有注册的 Scorer 成绩
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
                pass
        
        # 2. 智能化筛选最高级/最准确的一个版本展示给用户
        cleaned_results = {}
        
        # [A] 评估: 第一声开口延迟 (StartOffset/ALAL)
        so_key = "StartOffset_CA" if computation_aware else "StartOffset"
        align_so_key = "StartOffset_SpeechAlign_CA" if computation_aware else "StartOffset_SpeechAlign"
        
        if results.get(align_so_key) is not None and results[align_so_key] > 0:
            cleaned_results["First_Audio_Delay_(ALAL_ms)"] = results[align_so_key]
        else:
            cleaned_results["First_Audio_Delay_(ALAL_ms)"] = results.get(so_key, 0)
            
        # [B] 评估: 整句同传综合延迟 (标准版 ATD: 连同阅读/播放音频时间一起计算)
        atd_key = "ATD_CA" if computation_aware else "ATD"
        align_atd_key = "ATD_SpeechAlign_CA" if computation_aware else "ATD_SpeechAlign"
        
        if results.get(align_atd_key) is not None and results[align_atd_key] > 0:
            cleaned_results["Overall_Translation_Delay_(ATD_ms)"] = results[align_atd_key]
        else:
            cleaned_results["Overall_Translation_Delay_(ATD_ms)"] = results.get(atd_key, 0)

        # [C] 评估: 模型结单同传延迟 (纯净版 CustomATD: 剔除音频合成自身的物理时长，只看模型推断何时结束)
        catd_key = "CustomATD_CA" if computation_aware else "CustomATD"
        align_catd_key = "CustomATD_SpeechAlign_CA" if computation_aware else "CustomATD_SpeechAlign"
        
        if results.get(align_catd_key) is not None and results[align_catd_key] > 0:
            cleaned_results["End_Action_Delay_(CustomATD_ms)"] = results[align_catd_key]
        else:
            cleaned_results["End_Action_Delay_(CustomATD_ms)"] = results.get(catd_key, 0)

        # [D] 评估: 实时率指标 (RTF)
        if "RTF" in results:
            cleaned_results["Real_Time_Factor_(RTF)"] = results["RTF"]
        elif "RTF_CA" in results:
            cleaned_results["Real_Time_Factor_(RTF)"] = results["RTF_CA"]

        # ================= 修改开始 =================
        # 如果用户显式开启了展示开关，则将原始杂乱数据装入 "detailed_all_metrics"
        if show_all_metrics:
            cleaned_results["detailed_all_metrics"] = results
        # ================= 修改结束 =================

        return cleaned_results

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
        # 注意: 对于 s2s, instances[i].prediction 应该是音频文件路径
        predictions = [ins.get_prediction_content() for ins in instances.values()]
        
        if args.task == "s2t":
            # [修改] 调用 TranslationEvaluator 算 BLEU
            # 默认只开 BLEU 以保证速度
            qual_eval = TranslationEvaluator(
                use_bleu=True, 
                use_chrf=False, 
                use_comet=False, 
                use_whisper=False
            )
            # 使用新的 evaluate_all 接口
            q_res = qual_eval.evaluate_all(
                target_text=predictions, 
                reference=ref, 
                source=src
            )
            print(q_res)
            
        elif args.task == "s2s":
            # [修改] 调用 TranslationEvaluator 替代原 SpeechEvaluator
            # 开启语音相关功能: use_wer=True, use_whisper=True
            # predictions 是 wav 路径列表
            speech_eval = TranslationEvaluator(
                use_wer=True, 
                use_whisper=True, 
                whisper_model="tiny", # 用 tiny 快速出 WER
                use_utmos=False       # 可选开启
            )
            # 使用 evaluate_all 接口传入 target_speech
            s_res = speech_eval.evaluate_all(
                target_speech=predictions, 
                reference=ref,
                source=src
            )
            print(s_res)

if __name__ == "__main__":
    main()