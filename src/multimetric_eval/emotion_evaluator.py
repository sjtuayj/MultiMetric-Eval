import os
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from tqdm import tqdm
from transformers import (
    AutoConfig, 
    AutoFeatureExtractor,
    Wav2Vec2FeatureExtractor, 
    AutoModelForAudioClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

def _load_data_list(
    input_data: Union[str, List[str]], 
    name: str,
    target_type: str = "audio"
) -> List[str]:
    if isinstance(input_data, list):
        return input_data
    if not isinstance(input_data, str):
        raise ValueError(f"{name} 必须是 文件路径(str) 或 列表(List[str])")

    path = Path(input_data)
    if not path.exists():
        raise FileNotFoundError(f"{name} 文件不存在: {input_data}")
    
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if target_type == "audio":
            candidate_keys = ["audio", "path", "file", "wav", "mp3"]
        elif target_type == "label":
            candidate_keys = ["label", "emotion", "class", "reference"]
        else:
            candidate_keys = ["text", "sentence", "content", "transcript"]

        if isinstance(data, list):
            if not data: return []
            if isinstance(data[0], str): return data
            if isinstance(data[0], dict):
                for key in candidate_keys:
                    if key in data[0]:
                        return [item[key] for item in data]
                raise ValueError(f"JSON 列表项中未找到常见{target_type}字段: {candidate_keys}")

        if isinstance(data, dict):
            plural_candidates = [k + "s" for k in candidate_keys]
            plural_candidates += ["target_text", "hypothesis", "source_text"] if target_type == "text" else []
            for key in plural_candidates:
                if key in data:
                    return data[key]
            raise ValueError(f"JSON 字典中未找到常见{target_type}列表字段")
        raise ValueError("不支持的 JSON 格式")
    else:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]


def _load_audio_from_folder(folder_path: str, extensions: tuple = (".wav", ".mp3", ".flac")) -> List[str]:
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


class EmotionEvaluator:
    """
    全能型情感评测器 
    支持：
    [1] 跨语种保真度评测 (Continuous V-A Benchmark)：比较两端音频在Arousal/Valence空间上的偏差。
    [2] 离散情感分类评测 (Classification Accuracy)：基于Hubert等多语种大模型的离散标签准确度评测。
    """

    # 保真度特征基座
    DEFAULT_AUDIO_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    DEFAULT_TEXT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    DEFAULT_WHISPER_MODEL = "large-v3"
    
    # 分类打分基座
    DEFAULT_CLS_MODEL = "superb/hubert-large-superb-er"

    def __init__(self, 
                 audio_model_path: Optional[str] = None,
                 text_model_path: Optional[str] = None,
                 whisper_model_path: Optional[str] = None,
                 cls_model_path: Optional[str] = None,
                 custom_label_map: Optional[Dict[str, str]] = None,
                 device: Optional[str] = None):
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.audio_model_path = audio_model_path or self.DEFAULT_AUDIO_MODEL
        self.text_model_path = text_model_path or self.DEFAULT_TEXT_MODEL
        self.whisper_model_path = whisper_model_path or self.DEFAULT_WHISPER_MODEL
        self.cls_model_path = cls_model_path or self.DEFAULT_CLS_MODEL
        
        # 内部状态
        self.audio_model = None
        self.audio_processor = None
        self.text_model = None
        self.text_tokenizer = None
        self.whisper_model = None
        self.cls_model = None
        self.cls_processor = None
        
        # 将用户的自定义标签映射统一全部小写，确保健壮性
        self.custom_label_map = {k.lower(): v.lower() for k, v in (custom_label_map or {}).items()}
        
        self._a_idx, self._d_idx, self._v_idx = 0, 1, 2

    # =============== 模块懒加截部分 ===============

    def _load_audio_model(self):
        if self.audio_model is not None: return
        print(f"⏳ 正在加载 V-A 连续特征提取模型: {self.audio_model_path}")
        try:
            self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(self.audio_model_path)
            self.audio_model = AutoModelForAudioClassification.from_pretrained(self.audio_model_path)
            self.audio_model.to(self.device).eval()
            if hasattr(self.audio_model.config, "id2label"):
                dim_indices = {v.lower(): int(k) for k, v in self.audio_model.config.id2label.items()}
                self._a_idx = dim_indices.get("arousal", 0)
                self._d_idx = dim_indices.get("dominance", 1)
                self._v_idx = dim_indices.get("valence", 2)
            print("✅ V-A 声学模型加载成功！")
        except Exception as e:
            print(f"❌ V-A 声学模型加载失败: {e}")

    def _load_text_model(self):
        if self.text_model is not None: return
        print(f"⏳ 正在加载文本情感模型 (Valence): {self.text_model_path}")
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_path)
            self.text_model = AutoModelForSequenceClassification.from_pretrained(self.text_model_path)
            self.text_model.to(self.device).eval()
            print("✅ 文本模型加载成功！")
        except Exception as e:
            print(f"❌ 文本模型加载失败: {e}")

    def _load_whisper_model(self):
        if self.whisper_model is not None: return
        print(f"⏳ 正在加载 Whisper ASR 模型: {self.whisper_model_path}")
        try:
            import whisper
            self.whisper_model = whisper.load_model(self.whisper_model_path, device=self.device)
            print("✅ Whisper 模型加载成功！")
        except ImportError:
            print("❌ Whisper 未安装，请使用 pip install openai-whisper 安装。")
        except Exception as e:
            print(f"❌ Whisper 模型加载失败: {e}")

    def _load_cls_model(self):
        if self.cls_model is not None: return
        print(f"⏳ 正在加载离散情感分类大模型: {self.cls_model_path}")
        try:
            self.cls_processor = AutoFeatureExtractor.from_pretrained(self.cls_model_path)
            self.cls_model = AutoModelForAudioClassification.from_pretrained(self.cls_model_path)
            self.cls_model.to(self.device).eval()
            print("✅ 离散分类大模型加载成功！")
        except Exception as e:
            print(f"❌ 离散分类模型加载失败: {e}")

    def _preprocess_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        target_sr = 16000
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)
            return waveform.squeeze()
        except Exception as e:
            print(f"⚠️ 读取音频失败 {audio_path}: {e}")
            return None

    # =============== 特征提取部分 ===============

    def _extract_audio_emotion(self, audio_paths: List[str]) -> List[Tuple[float, float, float]]:
        """提取 V-A 连续特征"""
        self._load_audio_model()
        if not self.audio_model: return [(0.0, 0.0, 0.0)] * len(audio_paths)

        results = []
        for path in tqdm(audio_paths, desc="🔍 音频 V-A 特征", unit="file"):
            waveform = self._preprocess_audio(path)
            if waveform is None:
                results.append((0.0, 0.0, 0.0))
                continue
            try:
                inputs = self.audio_processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self.audio_model(**inputs).logits[0]
                logits_np = logits.cpu().numpy()
                results.append((float(logits_np[self._a_idx]), float(logits_np[self._d_idx]), float(logits_np[self._v_idx])))
            except Exception as e:
                results.append((0.0, 0.0, 0.0))
        return results

    def _extract_cls_emotion(self, audio_paths: List[str]) -> List[str]:
        """提取离散情感分类预测结果"""
        self._load_cls_model()
        if not self.cls_model: return ["unknown"] * len(audio_paths)

        results = []
        for path in tqdm(audio_paths, desc="🔍 音频离散分类标签", unit="file"):
            waveform = self._preprocess_audio(path)
            if waveform is None:
                results.append("unknown")
                continue
            try:
                inputs = self.cls_processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self.cls_model(**inputs).logits[0]
                
                predicted_id = torch.argmax(logits, dim=-1).item()
                label = self.cls_model.config.id2label[predicted_id].lower()
                
                # 应用自定义对齐（如用户想把 'hap' 强转为 'happy'）
                if label in self.custom_label_map:
                    label = self.custom_label_map[label]
                results.append(label)
            except Exception as e:
                results.append("unknown")
        return results

    def _extract_text_emotion(self, texts: List[str]) -> List[float]:
        """提取文本 Valence 分数"""
        self._load_text_model()
        if not self.text_model: return [0.5] * len(texts)

        results = []
        for text in tqdm(texts, desc="🔍 文本 Valence 特征", unit="text"):
            if not text or not text.strip():
                results.append(0.5)
                continue
            try:
                inputs = self.text_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self.text_model(**inputs).logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                # 语义极性叠加 -> negative: 0, neutral: 0.5, positive: 1.0 (根据模型配置情况)
                valence = probs[2]*1.0 + probs[1]*0.5 + probs[0]*0.0
                results.append(float(valence))
            except Exception:
                results.append(0.5)
        return results

    def _transcribe_audio(self, audio_paths: List[str]) -> List[str]:
        """静默处理文本转写"""
        self._load_whisper_model()
        if not self.whisper_model: return [""] * len(audio_paths)
            
        results = []
        for path in tqdm(audio_paths, desc="🎙️ ASR 静默转写", unit="file"):
            try:
                res = self.whisper_model.transcribe(path, fp16=(self.device == "cuda"))
                results.append(res["text"].strip())
            except Exception:
                results.append("")
        return results

    # =============== 评测主入口 ===============

    def evaluate_all(self, 
                     source_audio: Optional[Union[List[str], str]] = None, 
                     target_audio: Optional[Union[List[str], str]] = None,
                     source_text: Optional[Union[List[str], str]] = None,
                     target_text: Optional[Union[List[str], str]] = None,
                     reference_labels: Optional[Union[List[str], str]] = None,
                     verbose: bool = True) -> Dict[str, float]:
        """
        动态融合引擎入口：你可以任选搭配参数调用。
        - 仅传 target_audio 与 reference_labels -> 计算 Accuracy 百分比打分
        - 仅传 source_audio 与 target_audio -> 计算 A/V 连续距离指标 (保真度)
        - 全部都传 -> 全面出具上述所有指标
        """
        
        # 智能参数判定适配
        if target_audio is None:
            # 如果使用者只传了第一个作为 source_audio 且附带了标签参数，我们允许将其智能 fallback 为分类音频
            if source_audio is not None and reference_labels is not None:
                target_audio = source_audio
                source_audio = None
            else:
                raise ValueError("🚨 必须提供 target_audio 参数运行评测 (保真度评测需同时提供 source_audio；分类评测需提供 reference_labels)。")

        # 规范化加载核心主体
        if isinstance(target_audio, str) and os.path.isdir(target_audio):
            tgt_paths = _load_audio_from_folder(target_audio)
        else:
            tgt_paths = _load_data_list(target_audio, "Target Audio Paths", "audio")
        
        num_samples = len(tgt_paths)
        if num_samples == 0:
            raise ValueError("没有找到目标音频数据，评测停止。")

        # 判断用户的评测意图
        run_fidelity = source_audio is not None
        run_classification = reference_labels is not None
        results = {}

        # ==================== 分支 1：保真度/多模态距离计算 ====================
        if run_fidelity:
            if verbose: print(f"\n📝 启动【保真度运算】 ({num_samples} 个 Source-Target 音频距离比对)...")
            
            # 源音频
            if isinstance(source_audio, str) and os.path.isdir(source_audio):
                src_paths = _load_audio_from_folder(source_audio)
            else:
                src_paths = _load_data_list(source_audio, "Source Audio Paths", "audio")

            if len(src_paths) != num_samples:
                raise ValueError(f"数目不一致: Source ({len(src_paths)}) != Target ({num_samples})")

            # 1. 音频特征提取
            src_audio_emotions = self._extract_audio_emotion(src_paths)
            tgt_audio_emotions = self._extract_audio_emotion(tgt_paths)

            # 2. 文本特征准备 (按需使用 ASR)
            src_texts = _load_data_list(source_text, "Source Texts", "text") if source_text else self._transcribe_audio(src_paths)
            tgt_texts = _load_data_list(target_text, "Target Texts", "text") if target_text else self._transcribe_audio(tgt_paths)

            # 3. 文本极性抽取
            src_text_emotions = self._extract_text_emotion(src_texts)
            tgt_text_emotions = self._extract_text_emotion(tgt_texts)

            # 4. 融合与计算融合 MSE 方差
            w_a_audio = 1.0  
            w_v_audio = 0.3  
            w_v_text = 0.7   

            audio_a_dist_total, text_v_dist_total, fused_dist_total = 0.0, 0.0, 0.0
            
            for i in range(num_samples):
                # 源端组装
                s_audio_A, s_audio_D, s_audio_V = src_audio_emotions[i]
                s_final_A = s_audio_A * w_a_audio
                s_final_V = s_audio_V * w_v_audio + src_text_emotions[i] * w_v_text

                # 目标端组装
                t_audio_A, t_audio_D, t_audio_V = tgt_audio_emotions[i]
                t_final_A = t_audio_A * w_a_audio
                t_final_V = t_audio_V * w_v_audio + tgt_text_emotions[i] * w_v_text

                # 误差累计 MSE 与 欧氏距离
                audio_a_dist_total += (s_audio_A - t_audio_A) ** 2
                text_v_dist_total += (src_text_emotions[i] - tgt_text_emotions[i]) ** 2
                fused_dist_total += np.sqrt((s_final_A - t_final_A)**2 + (s_final_V - t_final_V)**2)

            results["Audio_Arousal_Distance_MSE"] = round(audio_a_dist_total / num_samples, 4)
            results["Text_Valence_Distance_MSE"] = round(text_v_dist_total / num_samples, 4)
            results["Final_Fused_Distance_Euclidean"] = round(fused_dist_total / num_samples, 4)


        # ==================== 分支 2：离散情感识别计算 ====================
        if run_classification:
            if verbose: print(f"\n📝 启动【分类准确率运算】 ({num_samples} 个特征识别与金标准比对)...")
            refs = _load_data_list(reference_labels, "Reference Labels", "label")
            
            if len(refs) != num_samples:
                raise ValueError(f"参考标签数量 ({len(refs)}) 与目标音频数量 ({num_samples}) 不匹配！")

            # 抽取目标预测类别
            preds = self._extract_cls_emotion(tgt_paths)
            
            # 百分比打分
            correct = 0
            for p, r in zip(preds, refs):
                # 均基于纯小写进行严格字符串对齐比较
                if p.strip() == r.strip().lower():
                    correct += 1
            
            acc = correct / len(refs) if len(refs) > 0 else 0.0
            results["Audio_Emotion_Accuracy"] = round(acc, 4)


        # ==================== 输出反馈 ====================
        if verbose:
            print("\n📊 [EmotionEvaluator] 多模态情感综合评测报告:")
            print(f"   - 有效评测样本量: {num_samples}条")
            
            if run_fidelity:
                print("   [维度保真度/损失度] (数值越小代表保真度越高)")
                print(f"   - Audio Arousal (声学唤醒度) 误差 MSE:         {results['Audio_Arousal_Distance_MSE']:.4f}")
                print(f"   - Text Valence (文本侧效价)  误差 MSE:         {results['Text_Valence_Distance_MSE']:.4f}")
                print(f"   - 跨模态综合特征融合欧氏距离 (Fused Distance): {results['Final_Fused_Distance_Euclidean']:.4f}")
            
            if run_classification:
                print("   [基座分类准确率] (数值越高代表识别正确率越高)")
                print(f"   - Audio Emotion Accuracy (离散情感识别准确率): {results['Audio_Emotion_Accuracy']:.2%}")
                
        return results
