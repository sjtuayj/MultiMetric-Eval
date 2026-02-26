"""
MultiMetric Eval - 多指标翻译评测工具
"""
import os
import gc
import json
import numpy as np
import sacrebleu
import torch
from typing import Dict, List, Optional, Union
from pathlib import Path

# ==================== 配置 ====================

CACHE_PATHS = {
    "huggingface": os.path.expanduser("~/.cache/huggingface/hub"),
    "whisper": os.path.expanduser("~/.cache/whisper"),
}

# ==================== 可选依赖 ====================

try:
    import whisper
except ImportError:
    whisper = None

try:
    from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
    HAS_BLEURT = True
except ImportError:
    HAS_BLEURT = False

try:
    from comet import download_model, load_from_checkpoint
except ImportError:
    download_model = None
    load_from_checkpoint = None


# ==================== 输入加载工具 ====================

def load_target_text_from_file(file_path: str) -> List[str]:
    """
    从文件加载翻译结果 (target_text)
    
    支持格式:
    - .json: 
        1. {"target_text": [...]}  (推荐)
        2. {"hypothesis": [...]}   (兼容旧格式)
        3. [{"target_text": "..."}, ...]
    - .txt: 每行一句
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    suffix = path.suffix.lower()
    
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 情况 1: 根目录是字典 (Dict)
        if isinstance(data, dict):
            # 优先找新名字
            if "target_text" in data:
                return data["target_text"]
            # 兼容旧名字
            if "hypothesis" in data:
                return data["hypothesis"]
            raise ValueError("JSON 中未找到 'target_text' 或 'hypothesis' 字段")
        
        # 情况 2: 根目录是列表 (List)
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            
            # 子项是字典: [{"target_text": "..."}]
            if isinstance(first_item, dict):
                if "target_text" in first_item:
                    return [item["target_text"] for item in data]
                if "hypothesis" in first_item:
                    return [item["hypothesis"] for item in data]
                raise ValueError("JSON 列表项中未找到 'target_text' 或 'hypothesis' 键")
            
            # 子项是字符串: ["你好", "世界"]
            if isinstance(first_item, str):
                return data
            
        raise ValueError("JSON 格式无法解析或为空")
    
    elif suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")


def load_audio_from_folder(folder_path: str, extensions: tuple = (".wav", ".mp3", ".flac")) -> List[str]:
    """从文件夹加载音频文件路径"""
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")
    
    audio_files = []
    for ext in extensions:
        audio_files.extend(folder.glob(f"*{ext}"))
    
    audio_files = sorted(audio_files, key=lambda x: x.stem)
    
    if not audio_files:
        raise ValueError(f"文件夹中没有音频文件: {folder_path}")
    
    return [str(f) for f in audio_files]


# ==================== 评测器 ====================

# BLEURT 默认模型（用于在线下载）
DEFAULT_BLEURT_MODEL = "lucadiliello/BLEURT-20"

class TranslationEvaluator:
    """多指标翻译评测器"""

    def __init__(self, use_comet=True, use_bleurt=False, use_whisper=False, 
                 comet_model="Unbabel/wmt22-comet-da", whisper_model="medium", 
                 bleurt_path=None, bleurt_model=None, device=None):
        """
        Args:
            use_comet: 是否启用 COMET
            use_bleurt: 是否启用 BLEURT
            use_whisper: 是否启用 Whisper
            comet_model: COMET 模型名称
            whisper_model: Whisper 模型名称
            bleurt_path: BLEURT 本地模型路径（优先使用）
            bleurt_model: BLEURT 在线模型名称（无本地路径时使用，默认 lucadiliello/BLEURT-20）
            device: 计算设备
        """
        # ===== 修改部分 START =====
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        if self.device.startswith("cuda"):
            if torch.cuda.is_available():
                if ":" in self.device:
                    gpu_id = int(self.device.split(":")[1])
                    gpu_name = torch.cuda.get_device_name(gpu_id)
                    print(f"🚀 初始化评测器 (设备: {self.device} - {gpu_name})")
                else:
                    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"🚀 初始化评测器 (设备: {self.device} - {gpu_name}, CUDA_VISIBLE_DEVICES={visible})")
            else:
                print(f"⚠️ 指定了 {self.device} 但 CUDA 不可用，回退到 CPU")
                self.device = "cpu"
                print(f"🚀 初始化评测器 (设备: cpu)")
        else:
            print(f"🚀 初始化评测器 (设备: {self.device})")
        # ===== 修改部分 END =====

        self.comet = self._load_comet(comet_model) if use_comet else None
        self.whisper_model = self._load_whisper(whisper_model) if use_whisper else None
        self.bleurt_model = None
        self.bleurt_tokenizer = None
        if use_bleurt:
            self._load_bleurt(bleurt_path, bleurt_model)

        print("✅ 系统就绪！")

    # -------------------- 上下文管理器 --------------------
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def cleanup(self):
        """释放模型显存"""
        if hasattr(self, 'comet') and self.comet is not None:
            del self.comet
            self.comet = None
        if hasattr(self, 'whisper_model') and self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
        if hasattr(self, 'bleurt_model') and self.bleurt_model is not None:
            del self.bleurt_model
            self.bleurt_model = None
        if hasattr(self, 'bleurt_tokenizer') and self.bleurt_tokenizer is not None:
            del self.bleurt_tokenizer
            self.bleurt_tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        print("🧹 已释放模型显存")

    # -------------------- 模型加载 --------------------

    def _load_comet(self, model_name: str):
        """加载 COMET 模型"""
        if not download_model:
            print("⚠️ COMET 未安装: pip install unbabel-comet")
            return None

        cache = os.path.join(CACHE_PATHS["huggingface"], f"models--{model_name.replace('/', '--')}")
        status = "[Local]" if os.path.exists(cache) else "[Online]"
        print(f"⏳ {status} 加载 COMET: {model_name}")

        model = load_from_checkpoint(download_model(model_name))
        model = model.to(self.device) if self.device == "cuda" else model
        print("✅ COMET 加载成功！")
        return model

    def _load_whisper(self, model_name: str):
        """加载 Whisper 模型"""
        if not whisper:
            print("⚠️ Whisper 未安装: pip install openai-whisper")
            return None

        cache = os.path.join(CACHE_PATHS["whisper"], f"{model_name}.pt")
        status = "[Local]" if os.path.exists(cache) else "[Online]"
        print(f"⏳ {status} 加载 Whisper: {model_name}")

        model = whisper.load_model(model_name, device=self.device)
        print("✅ Whisper 加载成功！")
        return model

    def _load_bleurt(self, path: Optional[str], model_name: Optional[str] = None):
        """加载 BLEURT PyTorch 模型（支持本地路径或在线下载）"""
        if not HAS_BLEURT:
            print("⚠️ bleurt-pytorch 未安装: pip install git+https://github.com/lucadiliello/bleurt-pytorch.git")
            return

        # 确定模型来源：本地路径优先，否则在线下载
        if path and os.path.exists(path):
            model_source = path
            status = "[Local]"
        elif path and not os.path.exists(path):
            print(f"⚠️ BLEURT 本地路径不存在: {path}，尝试在线下载...")
            model_source = model_name or DEFAULT_BLEURT_MODEL
            status = "[Online]"
        else:
            model_source = model_name or DEFAULT_BLEURT_MODEL
            # 检查 HuggingFace 缓存
            cache = os.path.join(
                CACHE_PATHS["huggingface"],
                f"models--{model_source.replace('/', '--')}"
            )
            status = "[Local]" if os.path.exists(cache) else "[Online]"

        print(f"⏳ {status} 加载 BLEURT: {model_source}")

        try:
            self.bleurt_tokenizer = BleurtTokenizer.from_pretrained(model_source)
            self.bleurt_model = BleurtForSequenceClassification.from_pretrained(model_source)
            self.bleurt_model = self.bleurt_model.to(self.device)
            self.bleurt_model.eval()
            print("✅ BLEURT 加载成功！")
        except Exception as e:
            print(f"⚠️ BLEURT 加载失败: {e}")
            self.bleurt_model = None
            self.bleurt_tokenizer = None

    def _compute_bleurt(self, references: List[str], candidates: List[str]) -> float:
        """使用 BLEURT PyTorch 模型计算分数"""
        all_scores = []
        batch_size = 32

        for i in range(0, len(references), batch_size):
            batch_refs = references[i:i + batch_size]
            batch_cands = candidates[i:i + batch_size]

            with torch.no_grad():
                inputs = self.bleurt_tokenizer(
                    batch_refs, batch_cands,
                    padding='longest',
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                scores = self.bleurt_model(**inputs).logits.flatten().tolist()
                all_scores.extend(scores)

        return float(np.mean(all_scores))

    # -------------------- 核心内部方法 --------------------

    def transcribe(self, audio_paths: List[str]) -> List[str]:
        """语音转文字"""
        if not self.whisper_model:
            raise RuntimeError("请设置 use_whisper=True")

        print(f"🎤 ASR 转写 ({len(audio_paths)} 个文件)...")
        results = []
        for i, path in enumerate(audio_paths, 1):
            if not os.path.exists(path):
                print(f"   ⚠️ [{i}] 文件不存在: {path}")
                results.append("")
            else:
                try:
                    text = self.whisper_model.transcribe(path, fp16=(self.device == "cuda"))["text"]
                    results.append(text.strip())
                    print(f"   ✓ [{i}/{len(audio_paths)}] {os.path.basename(path)}")
                except Exception as e:
                    print(f"   ⚠️ [{i}] 转写失败: {e}")
                    results.append("")
        return results

    def _compute_metrics(
        self,
        target_text: List[str],  # <--- 改名了
        reference: List[str],
        source: Optional[List[str]] = None,
        suffix: str = "",
    ) -> Dict[str, float]:
        """核心计算逻辑"""
        results = {}

        # SacreBLEU (使用 target_text)
        results[f"sacreBLEU{suffix}"] = self._safe_calc(
            lambda: sacrebleu.corpus_bleu(target_text, [reference]).score
        )

        results[f"chrF++{suffix}"] = self._safe_calc(
            lambda: sacrebleu.corpus_chrf(target_text, [reference], word_order=2).score
        )

        # BLEURT (使用 target_text)
        if self.bleurt_model and self.bleurt_tokenizer:
            results[f"BLEURT{suffix}"] = self._safe_calc(
                lambda: self._compute_bleurt(reference, target_text)
            )

        # COMET (使用 target_text)
        if self.comet:
            if source:
                # 注意这里字典的 key: "mt" 对应 target_text
                data = [{"src": s, "mt": t, "ref": r} for s, t, r in zip(source, target_text, reference)]
                gpus = 1 if self.device.startswith("cuda") else 0
                results[f"COMET{suffix}"] = self._safe_calc(
                    lambda: self.comet.predict(data, batch_size=8, gpus=gpus).system_score
                )
            else:
                results[f"COMET{suffix}"] = -1.0

        results = {k: round(v, 4) if v >= 0 else v for k, v in results.items()}
        return results

    # -------------------- 公开 API --------------------

    def evaluate(
        self,
        reference: List[str],
        source: Optional[List[str]] = None,
        target_text: Optional[Union[List[str], str]] = None, # <--- 改名了，支持 List 或 文件路径
        target_speech: Optional[Union[List[str], str]] = None, # 支持 List 或 文件夹路径
    ) -> Dict[str, Union[float, List[str]]]:
        """
        全能评测接口
        
        Args:
            reference: 参考译文列表
            source: 源文本列表 (COMET 需要)
            target_text: 翻译文本结果 (可以是 字符串列表 或 文件路径)
            target_speech: 翻译音频结果 (可以是 文件夹路径 或 音频路径列表)
        """
        if target_text is None and target_speech is None:
            raise ValueError("请至少提供 target_text 或 target_speech 之一")

        results = {}

        # 1. 处理文本评测
        if target_text is not None:
            # 自动判断是文件路径还是列表
            if isinstance(target_text, str):
                print(f"📂 加载翻译文件: {target_text}")
                # 假设 load_target_text_from_file 已经重命名好了
                final_text = load_target_text_from_file(target_text) 
            else:
                final_text = target_text

            if len(final_text) != len(reference):
                raise ValueError(f"文本数量不一致: target={len(final_text)}, ref={len(reference)}")

            print("📊 计算文本指标...")
            # 调用核心逻辑
            metrics = self._compute_metrics(final_text, reference, source, suffix="")
            results.update(metrics)
            results["target_text"] = final_text # 返回内容方便查看

        # 2. 处理语音评测 (ASR)
        if target_speech is not None:
            if not self.whisper_model:
                raise RuntimeError("需开启 use_whisper=True")
            
            # 自动判断是文件夹还是列表
            if isinstance(target_speech, str):
                print(f"📂 加载音频文件夹: {target_speech}")
                audio_paths = load_audio_from_folder(target_speech)
            else:
                audio_paths = target_speech

            if len(audio_paths) != len(reference):
                raise ValueError(f"音频数量不一致: audio={len(audio_paths)}, ref={len(reference)}")

            # 转录
            asr_text = self.transcribe(audio_paths)

            print("📊 计算 ASR 指标...")
            # 调用核心逻辑
            asr_metrics = self._compute_metrics(asr_text, reference, source, suffix="_ASR")
            results.update(asr_metrics)
            results["target_text_ASR"] = asr_text # 统一命名

        return results

    def evaluate_dataset(
        self,
        dataset,
        target_text: Optional[Union[List[str], str]] = None,
        target_speech: Optional[str] = None,
    ) -> Dict[str, Union[float, List[str]]]:
        """Dataset 辅助接口"""
        return self.evaluate(
            reference=dataset.reference_texts,
            source=dataset.source_texts,
            target_text=target_text,
            target_speech=target_speech,
        )

    # -------------------- 工具方法 --------------------

    @staticmethod
    def _safe_calc(fn, default=-1.0) -> float:
        try:
            return fn()
        except Exception as e:
            # ================= 新增：打印报错详情 =================
            print(f"\n❌ 计算出错: {e}")
            import traceback
            traceback.print_exc()
            # ===================================================
            return default