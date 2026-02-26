"""
SpeechEvaluator - 语音合成/转换质量评测工具
支持指标: 
1. UTMOS (Naturalness/Quality) - 只需要音频
2. WER/CER (Word/Character Error Rate) - 自动根据语言特性适配
"""

import os
import torch
import torchaudio
import pandas as pd
import jiwer
import re
from tqdm import tqdm
from pathlib import Path
from typing import List, Union, Optional, Dict

class SpeechEvaluator:
    def __init__(self, 
                 device: Optional[str] = None, 
                 whisper_model_name: str = "medium",
                 use_gpu: bool = True, 
                 utmos_model_path: Optional[str] = None,
                 utmos_ckpt_path: Optional[str] = None, # <--- 新增：指定权重路径
                 whisper_root: Optional[str] = None):
        """
        初始化评测器
        
        Args:
            device: 指定设备 (e.g. "cuda", "cuda:0", "cpu")。
            whisper_model_name: Whisper 模型大小
            use_gpu: (当 device 为 None 时生效) 是否优先使用 GPU
            utmos_model_path: UTMOS 本地源码路径 (包含 hubconf.py 的文件夹)
            utmos_ckpt_path: UTMOS 本地权重路径 (.pth 文件)
            whisper_root: Whisper 权重存放路径
        """
        
        # ================= 设备初始化逻辑 =================
        if device is not None:
            self.device = device
        elif torch.cuda.is_available() and use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # 详细的设备检查与日志打印
        if self.device.startswith("cuda"):
            if torch.cuda.is_available():
                if ":" in self.device:
                    try:
                        gpu_id = int(self.device.split(":")[1])
                        gpu_name = torch.cuda.get_device_name(gpu_id)
                        print(f"🚀 初始化语音评测器 (设备: {self.device} - {gpu_name})")
                    except:
                        print(f"🚀 初始化语音评测器 (设备: {self.device})")
                else:
                    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"🚀 初始化语音评测器 (设备: {self.device} - {gpu_name}, CUDA_VISIBLE_DEVICES={visible})")
            else:
                print(f"⚠️ 指定了 {self.device} 但 CUDA 不可用，回退到 CPU")
                self.device = "cpu"
                print(f"🚀 初始化语音评测器 (设备: cpu)")
        else:
            print(f"🚀 初始化语音评测器 (设备: {self.device})")
        # =================================================

        # 2. 加载 UTMOS 模型
        print("⏳ Loading UTMOS model...")
        try:
            # 确定源码来源 (Local dir or Github)
            source = "github"
            repo_or_dir = "tarepan/SpeechMOS"
            
            if utmos_model_path and os.path.isdir(utmos_model_path):
                source = "local"
                repo_or_dir = utmos_model_path
                print(f"   📂 Loading UTMOS code from: {repo_or_dir}")
            else:
                print("   🌐 Downloading UTMOS code from GitHub (tarepan/SpeechMOS)...")

            # 确定加载方式 (Pretrained or Custom Weights)
            # 如果用户提供了 ckpt 路径，我们就不让 hub 去下载默认权重 (pretrained=False)
            load_pretrained = (utmos_ckpt_path is None)

            self.utmos_model = torch.hub.load(
                repo_or_dir, 
                "utmos22_strong", 
                source=source, 
                trust_repo=True,
                pretrained=load_pretrained
            )
            
            # 手动加载权重
            if utmos_ckpt_path:
                if os.path.exists(utmos_ckpt_path):
                    print(f"   ⚖️ Loading UTMOS weights from: {utmos_ckpt_path}")
                    state_dict = torch.load(utmos_ckpt_path, map_location=self.device)
                    self.utmos_model.load_state_dict(state_dict)
                else:
                    print(f"⚠️ Warning: Checkpoint path not found: {utmos_ckpt_path}")
                    print("   Falling back to random initialization (Scores will be wrong!)")

            self.utmos_model.to(self.device)
            self.utmos_model.eval()
        except Exception as e:
            print(f"❌ Error loading UTMOS: {e}")
            raise e

        # 3. 加载 Whisper 模型
        self.whisper_model = None
        self.whisper_available = False
        
        # 3. 加载 Whisper 模型
        self.whisper_model = None
        self.whisper_available = False
        
        if whisper_model_name:  # <--- 新增：只有当提供了模型名时才尝试加载
            try:
                import whisper
                print(f"⏳ Loading Whisper model ({whisper_model_name})...")
                
                if whisper_root:
                    print(f"   📂 Cache dir: {whisper_root}")
                    if not os.path.exists(whisper_root):
                        os.makedirs(whisper_root, exist_ok=True)
                
                self.whisper_model = whisper.load_model(
                    whisper_model_name, 
                    device=self.device, 
                    download_root=whisper_root
                )
                self.whisper_available = True
            except ImportError:
                print("⚠️ Warning: 'openai-whisper' not installed. WER calculation will be disabled.")
            except Exception as e:
                print(f"❌ Error loading Whisper: {e}")
        else:
            print("ℹ️ Whisper model disabled (name is None). WER calculation will be disabled.")

        # 4. WER 基础文本清洗
        self.wer_transform = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemoveEmptyStrings(),
        ])
        
        print("✅ 系统就绪！")

    def _compute_utmos(self, audio_paths: List[str]) -> List[float]:
        scores = []
        target_sr = 16000 
        
        for path in tqdm(audio_paths, desc="computing UTMOS"):
            if not os.path.exists(path):
                scores.append(1.0)
                continue
            try:
                wave, sr = torchaudio.load(path)
                if sr != target_sr:
                    wave = torchaudio.functional.resample(wave, sr, target_sr)
                if wave.shape[0] > 1:
                    wave = torch.mean(wave, dim=0, keepdim=True)
                
                wave = wave.to(self.device)
                with torch.no_grad():
                    score = self.utmos_model(wave, target_sr)
                    scores.append(score.item())
            except Exception as e:
                print(f"⚠️ Error processing {path}: {e}")
                scores.append(1.0)
        return scores

    def _transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        if not self.whisper_available or self.whisper_model is None:
            raise RuntimeError("Whisper is not installed.")
        hyps = []
        for path in tqdm(audio_paths, desc="ASR Transcribing"):
            if not os.path.exists(path):
                hyps.append("")
                continue
            try:
                result = self.whisper_model.transcribe(path, fp16=False)
                text = result['text'].strip()
                hyps.append(text)
            except Exception as e:
                print(f"⚠️ ASR Error {path}: {e}")
                hyps.append("")
        return hyps

    def _smart_tokenize(self, text: str) -> str:
        """
        智能分词：
        1. 对中日韩(CJK)字符，在每个字符间插入空格 -> 实现 CER 计算
        2. 对印欧语系(英文等)，保持原有空格分隔 -> 实现 WER 计算
        """
        cjk_pattern = re.compile(r'([\u4e00-\u9fff])')
        text = cjk_pattern.sub(r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def evaluate(self, audio_paths: List[str], reference_texts: List[str]) -> Dict[str, float]:
        if len(audio_paths) != len(reference_texts):
            raise ValueError(f"数据量不匹配: 音频 {len(audio_paths)} vs 文本 {len(reference_texts)}")
        
        results = {}

        if hasattr(self, 'utmos_model'):
            print("\n--- 正在计算 UTMOS (自然度) ---")
            utmos_scores = self._compute_utmos(audio_paths)
            results['UTMOS'] = sum(utmos_scores) / len(utmos_scores) if utmos_scores else 0.0
            results.setdefault('details', {})['utmos_scores'] = utmos_scores

        if self.whisper_available:
            print("\n--- 正在计算 WER/CER (准确率) ---")
            hyp_texts = self._transcribe_batch(audio_paths)
            
            clean_refs = []
            clean_hyps = []

            for ref, hyp in zip(reference_texts, hyp_texts):
                r = self.wer_transform(ref)
                h = self.wer_transform(hyp)
                r = self._smart_tokenize(r)
                h = self._smart_tokenize(h)
                clean_refs.append(r)
                clean_hyps.append(h)
            
            error_rate = jiwer.wer(clean_refs, clean_hyps)
            results['WER'] = error_rate
            
            results.setdefault('details', {})['audio_paths'] = audio_paths
            results['details']['ref_texts'] = clean_refs
            results['details']['hyp_texts'] = clean_hyps
        
        return results

    def evaluate_from_csv(self, csv_path: str, wav_col: str = "wav_path", text_col: str = "text"):
        print(f"📂 从 CSV 加载数据: {csv_path}")
        df = pd.read_csv(csv_path)
        return self.evaluate(
            audio_paths=df[wav_col].tolist(),
            reference_texts=df[text_col].astype(str).tolist()
        )

    def evaluate_from_folder(self, audio_folder: str, text_file: str, audio_ext: str = ".wav"):
        print(f"📂 从文件夹加载音频: {audio_folder}")
        folder_path = Path(audio_folder)
        audio_paths = sorted([str(p) for p in folder_path.glob(f"*{audio_ext}")])
        print(f"📄 读取参考文本: {text_file}")
        with open(text_file, 'r', encoding='utf-8') as f:
            refs = [line.strip() for line in f if line.strip()]
        
        if len(audio_paths) != len(refs):
            print(f"⚠️ 警告: 音频文件数量 ({len(audio_paths)}) 与 文本行数 ({len(refs)}) 不一致！")
            min_len = min(len(audio_paths), len(refs))
            audio_paths = audio_paths[:min_len]
            refs = refs[:min_len]
        return self.evaluate(audio_paths, refs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Speech Quality & Accuracy Evaluator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", type=str, help="Path to metadata CSV file")
    group.add_argument("--folder", type=str, help="Path to audio folder")
    parser.add_argument("--text_file", type=str, help="Path to text file (required if using --folder)")
    
    # 路径参数
    parser.add_argument("--utmos_path", type=str, default=None, help="Local path to UTMOS source code (repo)")
    parser.add_argument("--utmos_ckpt", type=str, default=None, help="Local path to UTMOS weights (.pth)") # 新增
    parser.add_argument("--whisper_root", type=str, default=None, help="Local path to Whisper model weights")
    
    # === 支持各种 CUDA 指定方式 ===
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. 'cuda', 'cuda:0', 'cpu')")
    
    args = parser.parse_args()
    
    evaluator = SpeechEvaluator(
        device=args.device,
        utmos_model_path=args.utmos_path,
        utmos_ckpt_path=args.utmos_ckpt, # 传入新参数
        whisper_root=args.whisper_root
    )
    
    if args.csv:
        results = evaluator.evaluate_from_csv(args.csv)
    elif args.folder:
        results = evaluator.evaluate_from_folder(args.folder, args.text_file)

    print("\nEvaluation Report:")
    print(f"UTMOS: {results.get('UTMOS', 'N/A')}")
    print(f"WER/CER: {results.get('WER', 'N/A')}")