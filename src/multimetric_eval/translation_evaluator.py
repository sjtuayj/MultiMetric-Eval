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

def load_hypothesis_from_file(file_path: str) -> List[str]:
    """
    从文件加载用户翻译结果
    
    支持格式:
    - .json: {"hypothesis": [...]} 或 [{"id": "x", "hypothesis": "..."}, ...]
    - .txt: 每行一句
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    suffix = path.suffix.lower()
    
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "hypothesis" in data:
            return data["hypothesis"]
        
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and "hypothesis" in data[0]:
                return [item["hypothesis"] for item in data]
            if isinstance(data[0], str):
                return data
        
        raise ValueError("JSON 格式不正确")
    
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

    def __init__(
        self,
        use_comet: bool = True,
        use_bleurt: bool = False,
        use_whisper: bool = False,
        comet_model: str = "Unbabel/wmt22-comet-da",
        whisper_model: str = "medium",
        bleurt_path: Optional[str] = None,
        bleurt_model: Optional[str] = None,
        device: Optional[str] = None,
    ):
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
        hypothesis: List[str],
        reference: List[str],
        source: Optional[List[str]] = None,
        suffix: str = "",
    ) -> Dict[str, float]:

        results = {}

        results[f"sacreBLEU{suffix}"] = self._safe_calc(
            lambda: sacrebleu.corpus_bleu(hypothesis, [reference]).score
        )

        results[f"chrF++{suffix}"] = self._safe_calc(
            lambda: sacrebleu.corpus_chrf(hypothesis, [reference], word_order=2).score
        )

        if self.bleurt_model and self.bleurt_tokenizer:
            results[f"BLEURT{suffix}"] = self._safe_calc(
                lambda: self._compute_bleurt(reference, hypothesis)
            )

        if self.comet:
            if source:
                data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(source, hypothesis, reference)]
                # ===== 修复这一行 =====
                gpus = 1 if self.device.startswith("cuda") else 0
                results[f"COMET{suffix}"] = self._safe_calc(
                    lambda: self.comet.predict(data, batch_size=8, gpus=gpus).system_score
                )
            else:
                print(f"⚠️ COMET{suffix} 需要 source 参数")
                results[f"COMET{suffix}"] = -1.0

        results = {k: round(v, 4) if v >= 0 else v for k, v in results.items()}
        return results

    # -------------------- 公开 API --------------------

    def evaluate(
        self,
        hypothesis: List[str],
        reference: List[str],
        source: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        纯文本评测（不涉及 ASR）
        
        Args:
            hypothesis: 用户翻译结果 (target_text)
            reference: 参考译文
            source: 源文本（COMET 需要）
        
        Returns:
            {"sacreBLEU": ..., "chrF++": ..., "COMET": ..., "BLEURT": ...}
        """
        print("📊 计算文本指标...")
        return self._compute_metrics(hypothesis, reference, source, suffix="")

    def evaluate_file(
        self,
        hypothesis_file: str,
        reference: List[str],
        source: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """从文件加载翻译结果并评测"""
        print(f"📂 加载翻译结果: {hypothesis_file}")
        hypothesis = load_hypothesis_from_file(hypothesis_file)
        print(f"   加载了 {len(hypothesis)} 条翻译")
        return self.evaluate(hypothesis, reference, source)

    def evaluate_audio_folder(
        self,
        audio_folder: str,
        reference: List[str],
        source: Optional[List[str]] = None,
    ) -> Dict[str, Union[float, List[str]]]:
        """从音频文件夹评测（纯 ASR 模式）"""
        print(f"📂 加载音频文件夹: {audio_folder}")
        audio_paths = load_audio_from_folder(audio_folder)
        print(f"   找到 {len(audio_paths)} 个音频文件")
        
        hypothesis_asr = self.transcribe(audio_paths)
        
        print("📊 计算 ASR 指标...")
        results = self._compute_metrics(hypothesis_asr, reference, source, suffix="_ASR")
        results["hypothesis_ASR"] = hypothesis_asr
        return results

    def evaluate_all(
        self,
        reference: List[str],
        source: Optional[List[str]] = None,
        target_text: Optional[Union[List[str], str]] = None,
        target_speech: Optional[str] = None,
    ) -> Dict[str, Union[float, List[str]]]:
        """
        统一评测接口（支持自定义数据）
        
        支持三种模式:
        1. 只传 target_text → 返回 sacreBLEU, chrF++, COMET, BLEURT
        2. 只传 target_speech → 返回 sacreBLEU_ASR, chrF++_ASR, COMET_ASR, BLEURT_ASR, hypothesis_ASR
        3. 同时传两者 → 返回全部指标
        
        Args:
            reference: 参考译文
            source: 源文本（COMET 需要）
            target_text: 文本翻译结果（List[str] 或文件路径）
            target_speech: 音频文件夹路径
        
        Returns:
            评测结果字典
        """
        if target_text is None and target_speech is None:
            raise ValueError("请至少提供 target_text 或 target_speech 之一")

        results = {}

        # ---- 文本评测 ----
        if target_text is not None:
            if isinstance(target_text, str):
                print(f"📂 加载翻译文件: {target_text}")
                hyp_text = load_hypothesis_from_file(target_text)
                print(f"   加载了 {len(hyp_text)} 条翻译")
            else:
                hyp_text = target_text

            assert len(hyp_text) == len(reference), \
                f"target_text 数量 ({len(hyp_text)}) 与 reference 数量 ({len(reference)}) 不一致"

            print("📊 计算文本指标...")
            text_results = self._compute_metrics(hyp_text, reference, source, suffix="")
            results.update(text_results)
            results["hypothesis_text"] = hyp_text

        # ---- 语音评测 ----
        if target_speech is not None:
            if not self.whisper_model:
                raise RuntimeError("使用 target_speech 需要设置 use_whisper=True")

            print(f"📂 加载音频文件夹: {target_speech}")
            audio_paths = load_audio_from_folder(target_speech)
            print(f"   找到 {len(audio_paths)} 个音频文件")

            assert len(audio_paths) == len(reference), \
                f"音频文件数量 ({len(audio_paths)}) 与 reference 数量 ({len(reference)}) 不一致"

            if target_text is not None:
                hyp_text_len = len(hyp_text) if isinstance(target_text, list) else len(load_hypothesis_from_file(target_text))
                assert len(audio_paths) == hyp_text_len, \
                    f"target_text 数量 ({hyp_text_len}) 与 target_speech 数量 ({len(audio_paths)}) 不一致，同时输入时必须是同一批样本"

            hyp_asr = self.transcribe(audio_paths)

            print("📊 计算 ASR 指标...")
            asr_results = self._compute_metrics(hyp_asr, reference, source, suffix="_ASR")
            results.update(asr_results)
            results["hypothesis_ASR"] = hyp_asr

        return results

    def evaluate_dataset(
        self,
        dataset,
        target_text: Optional[Union[List[str], str]] = None,
        target_speech: Optional[str] = None,
        hypothesis: Optional[Union[List[str], str]] = None,
        audio_folder: Optional[str] = None,
    ) -> Dict[str, Union[float, List[str]]]:
        """
        使用数据集评测（支持同时输入文本和语音）
        
        支持三种模式:
        1. 只传 target_text → 返回 sacreBLEU, chrF++, COMET, BLEURT
        2. 只传 target_speech → 返回 sacreBLEU_ASR, chrF++_ASR, COMET_ASR, BLEURT_ASR
        3. 同时传两者 → 返回全部指标
        
        Args:
            dataset: Dataset 对象
            target_text: 文本翻译结果（List[str] 或文件路径）
            target_speech: 音频文件夹路径
            hypothesis: [向后兼容] 等同于 target_text
            audio_folder: [向后兼容] 等同于 target_speech
        
        Returns:
            评测结果字典
        """
        if hypothesis is not None and target_text is None:
            target_text = hypothesis
        if audio_folder is not None and target_speech is None:
            target_speech = audio_folder

        return self.evaluate_all(
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