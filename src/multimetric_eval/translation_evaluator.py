"""
MultiMetric Eval - 全能翻译与语音评测工具 (Text + Speech)
"""
import os
import gc
import json
import re
import numpy as np
import sacrebleu
import torch
import torchaudio
import jiwer
from typing import Dict, List, Optional, Union
from pathlib import Path
from tqdm import tqdm

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

def load_text_from_file_or_list(input_data: Union[str, List[str]], name: str = "text") -> List[str]:
    """
    通用文本加载器：
    - 如果是 List[str]，直接返回
    - 如果是 str (路径)，读取文件
      - .json: 解析 JSON 结构
      - 其他后缀 (.txt, .devtest, .ref 等): 均按行读取纯文本
    """
    if isinstance(input_data, list):
        return input_data
    
    if not isinstance(input_data, str):
        raise ValueError(f"{name} 必须是 文件路径(str) 或 文本列表(List[str])")

    path = Path(input_data)
    if not path.exists():
        raise FileNotFoundError(f"{name} 文件不存在: {input_data}")
    
    suffix = path.suffix.lower()
    
    # === 分支 1: JSON 处理 ===
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 尝试智能解析常见的 JSON 结构
        if isinstance(data, list):
            if len(data) == 0: return []
            # ["text1", "text2"]
            if isinstance(data[0], str): return data
            # [{"target_text": "..."}, ...]
            if isinstance(data[0], dict):
                for key in ["target_text", "hypothesis", "text", "ref", "reference", "src", "source"]:
                    if key in data[0]:
                        return [item[key] for item in data]
                raise ValueError(f"JSON 列表项中未找到常见文本字段")

        if isinstance(data, dict):
            # {"target_text": [...]}
            for key in ["target_text", "hypothesis", "text", "ref", "reference", "src", "source"]:
                if key in data:
                    return data[key]
            raise ValueError("JSON 字典中未找到常见文本字段")
            
        raise ValueError("不支持的 JSON 格式")
    
    # === 分支 2: 纯文本处理 (默认) ===
    # 不再限制后缀必须是 .txt，允许 .devtest, .src, .ref 等任意文本文件
    else:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]


def load_audio_from_folder(folder_path: str, extensions: tuple = (".wav", ".mp3", ".flac")) -> List[str]:
    """从文件夹加载音频文件路径"""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")
    audio_files = []
    for ext in extensions:
        audio_files.extend(folder.glob(f"*{ext}"))
    # 按文件名排序，确保与文本对应
    audio_files = sorted(audio_files, key=lambda x: x.stem)
    if not audio_files:
        raise ValueError(f"文件夹中没有音频文件: {folder_path}")
    return [str(f) for f in audio_files]


# ==================== 评测器核心类 ====================

# BLEURT 默认模型
DEFAULT_BLEURT_MODEL = "lucadiliello/BLEURT-20"

class TranslationEvaluator:
    """
    全能评测器：支持 文本翻译 (BLEU, COMET...) 和 语音合成 (UTMOS, WER...)
    """

    def __init__(self, 
                 # === 指标开关 (Explicit Flags) ===
                 use_bleu: bool = True,
                 use_chrf: bool = True,
                 use_wer: bool = True,        # 控制 WER/CER
                 use_comet: bool = True,      # 控制 COMET
                 use_utmos: bool = False,     # 控制 UTMOS (默认False)
                 use_whisper: bool = False,   # 控制 ASR (开启后才能计算 _ASR 指标)
                 use_bleurt: bool = False,    # 控制 BLEURT
                 
                 # === 模型参数 ===
                 comet_model: str = "Unbabel/wmt22-comet-da",
                 whisper_model: str = "medium",
                 utmos_model_path: Optional[str] = None, # UTMOS 源码路径
                 utmos_ckpt_path: Optional[str] = None,  # UTMOS 权重路径
                 bleurt_path: Optional[str] = None,
                 bleurt_model: Optional[str] = None,
                 device: Optional[str] = None):
        
        # 1. 保存开关状态
        self.use_bleu = use_bleu
        self.use_chrf = use_chrf
        self.use_wer = use_wer
        self.use_comet = use_comet
        self.use_utmos = use_utmos
        self.use_whisper = use_whisper
        self.use_bleurt = use_bleurt

        # 2. 设备初始化
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # 详细设备日志
        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                gpu_id = int(self.device.split(":")[1]) if ":" in self.device else 0
                gpu_name = torch.cuda.get_device_name(gpu_id)
                print(f"🚀 初始化评测器 (设备: {self.device} - {gpu_name})")
            except:
                print(f"🚀 初始化评测器 (设备: {self.device})")
        else:
            print(f"🚀 初始化评测器 (设备: {self.device})")

        # 3. 加载模型 (按需加载)
        self.comet = None
        if self.use_comet:
            self.comet = self._load_comet(comet_model)

        self.whisper_model = None
        if self.use_whisper:
            self.whisper_model = self._load_whisper(whisper_model)

        self.utmos_model = None
        if self.use_utmos:
            self._load_utmos(utmos_model_path, utmos_ckpt_path)

        self.bleurt_model = None
        self.bleurt_tokenizer = None
        if self.use_bleurt:
            self._load_bleurt(bleurt_path, bleurt_model)

        # 4. WER 文本清洗工具
        if self.use_wer:
            self.wer_transform = jiwer.Compose([
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemoveEmptyStrings(),
            ])

        print("✅ 系统就绪！")

    # -------------------- 上下文管理 --------------------
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): 
        self.cleanup()
        return False

    def cleanup(self):
        """释放所有模型显存"""
        for attr in ['comet', 'whisper_model', 'utmos_model', 'bleurt_model', 'bleurt_tokenizer']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 已释放模型显存")

    # -------------------- 模型加载逻辑 --------------------

    def _load_comet(self, model_name: str):
        if not download_model:
            print("⚠️ COMET 未安装，跳过加载")
            return None
        try:
            cache = os.path.join(CACHE_PATHS["huggingface"], f"models--{model_name.replace('/', '--')}")
            status = "[Local]" if os.path.exists(cache) else "[Online]"
            print(f"⏳ {status} 加载 COMET: {model_name}")
            model = load_from_checkpoint(download_model(model_name))
            if self.device.startswith("cuda"):
                model = model.to(self.device)
            return model
        except Exception as e:
            print(f"❌ COMET 加载失败: {e}")
            return None

    def _load_whisper(self, model_name: str):
        if not whisper:
            print("⚠️ Whisper 未安装，跳过加载")
            return None
        try:
            print(f"⏳ 加载 Whisper: {model_name}")
            return whisper.load_model(model_name, device=self.device)
        except Exception as e:
            print(f"❌ Whisper 加载失败: {e}")
            return None

    def _load_utmos(self, model_path: Optional[str], ckpt_path: Optional[str]):
        print("⏳ Loading UTMOS model...")
        try:
            source = "github"
            repo_or_dir = "tarepan/SpeechMOS"
            if model_path and os.path.isdir(model_path):
                source = "local"
                repo_or_dir = model_path
                print(f"   📂 Source: {repo_or_dir}")
            else:
                print("   🌐 Source: GitHub (tarepan/SpeechMOS)")

            load_pretrained = (ckpt_path is None)
            self.utmos_model = torch.hub.load(
                repo_or_dir, "utmos22_strong", source=source, trust_repo=True, pretrained=load_pretrained
            )
            
            if ckpt_path:
                if os.path.exists(ckpt_path):
                    print(f"   ⚖️ Weights: {ckpt_path}")
                    state_dict = torch.load(ckpt_path, map_location=self.device)
                    self.utmos_model.load_state_dict(state_dict)
                else:
                    print(f"⚠️ Checkpoint not found: {ckpt_path}")

            self.utmos_model.to(self.device)
            self.utmos_model.eval()
        except Exception as e:
            print(f"❌ UTMOS 加载失败: {e}")
            self.utmos_model = None

    def _load_bleurt(self, path: Optional[str], model_name: Optional[str]):
        if not HAS_BLEURT:
            print("⚠️ bleurt-pytorch 未安装，跳过加载")
            return
        
        model_source = path if (path and os.path.exists(path)) else (model_name or DEFAULT_BLEURT_MODEL)
        print(f"⏳ 加载 BLEURT: {model_source}")
        
        try:
            self.bleurt_tokenizer = BleurtTokenizer.from_pretrained(model_source)
            self.bleurt_model = BleurtForSequenceClassification.from_pretrained(model_source)
            self.bleurt_model = self.bleurt_model.to(self.device)
            self.bleurt_model.eval()
        except Exception as e:
            print(f"❌ BLEURT 加载失败: {e}")

    # -------------------- 内部计算逻辑 --------------------

    def _get_bleu_tokenizer_name(self, lang: str) -> str:
        """根据目标语言代码返回 sacreBLEU 支持的 tokenizer 名称"""
        if lang == 'zh':
            return 'zh'  # 使用 jieba
        elif lang == 'ja':
            return 'ja-mecab'  # 使用 MeCab
        elif lang == 'ko':
            return 'ko-mecab'  # 使用 MeCab-ko
        else:
            return '13a'  # 默认使用 WMT 标准分词

    def _transcribe(self, audio_paths: List[str]) -> List[str]:
        """Whisper 转写"""
        if not self.whisper_model:
            return [""] * len(audio_paths)
        
        print(f"🎤 ASR 转写中 ({len(audio_paths)} 个文件)...")
        results = []
        for path in tqdm(audio_paths, desc="Whisper Transcribing"):
            if not os.path.exists(path):
                results.append("")
                continue
            try:
                res = self.whisper_model.transcribe(path, fp16=(self.device == "cuda"))
                results.append(res["text"].strip())
            except Exception as e:
                print(f"⚠️ 转写出错 {path}: {e}")
                results.append("")
        return results

    def _preprocess_for_wer(self, text: str, lang: str) -> str:
        """
        WER/CER 计算前的预处理
        如果是中日韩语言 (CJK)，按字符切分（计算 CER）。
        如果是其他语言，保持原样（计算 WER）。
        """
        # 1. 基础清洗 (小写, 去标点)
        text = self.wer_transform(text)
        
        # 2. CJK 字符级切分
        if lang in ['zh', 'ja', 'ko']:
            # 去除可能存在的原有空格，然后按字插入空格
            text = text.replace(" ", "")
            return " ".join(list(text))
        
        return text

    def _compute_utmos_score(self, audio_paths: List[str]) -> float:
        """计算 UTMOS 平均分"""
        if not self.utmos_model: return -1.0
        
        scores = []
        target_sr = 16000
        for path in tqdm(audio_paths, desc="Calculating UTMOS"):
            if not os.path.exists(path):
                continue
            try:
                wave, sr = torchaudio.load(path)
                if sr != target_sr:
                    wave = torchaudio.functional.resample(wave, sr, target_sr)
                if wave.shape[0] > 1:
                    wave = torch.mean(wave, dim=0, keepdim=True)
                
                wave = wave.to(self.device)
                with torch.no_grad():
                    s = self.utmos_model(wave, target_sr)
                    scores.append(s.item())
            except Exception as e:
                print(f"⚠️ UTMOS Error {path}: {e}")
        
        return sum(scores) / len(scores) if scores else 0.0

    def _compute_wer_score(self, references: List[str], hypotheses: List[str], lang: str) -> Dict[str, float]:
        """计算 WER 或 CER"""
        clean_refs = []
        clean_hyps = []
        
        # 应用针对性的预处理
        for r, h in zip(references, hypotheses):
            clean_refs.append(self._preprocess_for_wer(r, lang))
            clean_hyps.append(self._preprocess_for_wer(h, lang))
        
        if not clean_refs: return {"WER": 0.0}
        
        error_rate = jiwer.wer(clean_refs, clean_hyps)
        
        # 动态命名指标
        metric_name = "CER" if lang in ['zh', 'ja', 'ko'] else "WER"
        return {metric_name: error_rate}

    def _compute_bleurt_score(self, references: List[str], candidates: List[str]) -> float:
        all_scores = []
        batch_size = 32
        for i in range(0, len(references), batch_size):
            br = references[i:i+batch_size]
            bc = candidates[i:i+batch_size]
            with torch.no_grad():
                inputs = self.bleurt_tokenizer(br, bc, padding='longest', truncation=True, max_length=512, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                scores = self.bleurt_model(**inputs).logits.flatten().tolist()
                all_scores.extend(scores)
        return float(np.mean(all_scores))

    def _compute_text_metrics_internal(self, target: List[str], ref: List[str], src: Optional[List[str]], suffix: str, lang: str) -> Dict:
        """通用文本指标计算 (BLEU, chrF, COMET, BLEURT)"""
        res = {}
        
        # 1. sacreBLEU (智能选择 tokenizer)
        if self.use_bleu:
            tokenizer_name = self._get_bleu_tokenizer_name(lang)
            try:
                # [修正] 参数名应该是 tokenize，不是 tokenizer
                res[f"sacreBLEU{suffix}"] = sacrebleu.corpus_bleu(target, [ref], tokenize=tokenizer_name).score
            except Exception as e:
                print(f"BLEU Error ({tokenizer_name}): {e}")
                res[f"sacreBLEU{suffix}"] = -1.0

        # 2. chrF++
        if self.use_chrf:
            try:
                res[f"chrF++{suffix}"] = sacrebleu.corpus_chrf(target, [ref], word_order=2).score
            except: res[f"chrF++{suffix}"] = -1.0

        # 3. BLEURT
        if self.use_bleurt and self.bleurt_model:
            try:
                res[f"BLEURT{suffix}"] = self._compute_bleurt_score(ref, target)
            except Exception as e:
                print(f"BLEURT Calc Error: {e}")
                res[f"BLEURT{suffix}"] = -1.0

        # 4. COMET
        if self.use_comet and self.comet and src:
            try:
                data = [{"src": s, "mt": t, "ref": r} for s, t, r in zip(src, target, ref)]
                gpus = 1 if self.device.startswith("cuda") else 0
                res[f"COMET{suffix}"] = self.comet.predict(data, batch_size=8, gpus=gpus).system_score
            except Exception as e:
                print(f"COMET Calc Error: {e}")
                res[f"COMET{suffix}"] = -1.0
        
        # 格式化小数
        return {k: round(v, 4) if v >= 0 else v for k, v in res.items()}

    # -------------------- 公开接口 --------------------

    def evaluate_all(
        self,
        reference: Union[List[str], str],
        source: Optional[Union[List[str], str]] = None,
        target_text: Optional[Union[List[str], str]] = None,
        target_speech: Optional[Union[List[str], str]] = None,
        src_speech: Optional[Union[List[str], str]] = None, # 占位符
        target_lang: str = "en"  # <--- [新增] 默认英语，支持 "zh", "ja", "ko"
    ) -> Dict[str, float]:
        """
        全能评测入口
        
        Args:
            reference: 参考文本 (List 或 文件路径)
            source: 源文本 (List 或 文件路径)
            target_text: 翻译文本 (List 或 文件路径)
            target_speech: 翻译语音 (List 或 文件夹路径)
            src_speech: 源语音 (保留参数)
            target_lang: 目标语言代码 ("en", "zh", "ja", "ko")，影响 BLEU 分词和 WER/CER 选择
        """
        results = {}
        print(f"\n--- 开始评测 (Target Lang: {target_lang}) ---")

        # 1. 加载并规范化文本输入
        try:
            final_ref = load_text_from_file_or_list(reference, "Reference")
            
            final_src = None
            if source:
                final_src = load_text_from_file_or_list(source, "Source")
                if len(final_src) != len(final_ref):
                    raise ValueError(f"Source ({len(final_src)}) 与 Reference ({len(final_ref)}) 数量不一致")
            
            final_text = None
            if target_text:
                final_text = load_text_from_file_or_list(target_text, "Target Text")
                if len(final_text) != len(final_ref):
                    raise ValueError(f"Target Text ({len(final_text)}) 与 Reference ({len(final_ref)}) 数量不一致")
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            raise e

        # 2. 加载语音输入
        final_audio_paths = None
        if target_speech:
            try:
                if isinstance(target_speech, str):
                    print(f"📂 加载音频文件夹: {target_speech}")
                    final_audio_paths = load_audio_from_folder(target_speech)
                else:
                    final_audio_paths = target_speech
                
                if len(final_audio_paths) != len(final_ref):
                    raise ValueError(f"音频数量 ({len(final_audio_paths)}) 与 Reference ({len(final_ref)}) 不一致")
            except Exception as e:
                print(f"❌ 音频加载失败: {e}")
                raise e

        if not final_text and not final_audio_paths:
            raise ValueError("必须提供 target_text 或 target_speech 其中之一")

        # ================= A. 纯文本评测 (Standard Metrics) =================
        if final_text:
            print(f"📄 检测到 target_text ({len(final_text)} lines)，计算标准文本指标...")
            text_metrics = self._compute_text_metrics_internal(
                target=final_text, 
                ref=final_ref, 
                src=final_src, 
                suffix="", # 无后缀
                lang=target_lang # 传入语言
            )
            results.update(text_metrics)

        # ================= B. 语音评测 (Speech & ASR Metrics) =================
        if final_audio_paths:
            print(f"🔊 检测到 target_speech ({len(final_audio_paths)} clips)...")
            
            # 1. UTMOS (自然度)
            if self.use_utmos:
                if self.utmos_model:
                    print("   ➤ 计算 UTMOS...")
                    results["UTMOS"] = self._compute_utmos_score(final_audio_paths)
                else:
                    print("   ⚠️ use_utmos=True 但模型未加载，跳过")

            # 2. ASR 相关 (Whisper 依赖)
            if self.use_whisper:
                if self.whisper_model:
                    # (1) 转写
                    asr_text = self._transcribe(final_audio_paths)
                    
                    # (2) WER/CER (准确率)
                    if self.use_wer:
                        print(f"   ➤ 计算 WER/CER (Mode: {target_lang})...")
                        wer_metrics = self._compute_wer_score(final_ref, asr_text, lang=target_lang)
                        results.update(wer_metrics)

                    # (3) ASR 后缀指标 (BLEU_ASR, COMET_ASR...)
                    print("   ➤ 计算 ASR 文本指标 (BLEU_ASR 等)...")
                    asr_metrics = self._compute_text_metrics_internal(
                        target=asr_text,
                        ref=final_ref,
                        src=final_src,
                        suffix="_ASR", # 强制后缀
                        lang=target_lang # 传入语言
                    )
                    results.update(asr_metrics)
                else:
                    print("   ⚠️ use_whisper=True 但模型未加载，跳过 ASR 相关指标")

        return results