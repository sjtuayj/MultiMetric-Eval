import os
import json
import torch
import librosa
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from tqdm import tqdm
from numpy.linalg import norm

def _load_data_list(
    input_data: Union[str, List[str]], 
    name: str,
    target_type: str = "audio"
) -> List[str]:
    # 与 EmotionEvaluator 相同的输入处理逻辑，确保兼容
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
            candidate_keys = ["label", "emotion", "class", "reference", "events"]
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


class ParalinguisticEvaluator:
    """
    副语言与声学事件评测器
    旨在评估 S2ST 模型（或其它声音转换任务）在翻译后能否较好地保留原始存在的非语言音频事件（如叹气、咳嗽、笑声等）。
    
    默认开启双轨测量：
    [1] 连续特征保真度 (Continuous Fidelity) - 基于 CLAP 取高维音频表征进行 Cosine 相似度计算（不依赖文本/时间）
    [2] 离散事件匹配度 (Discrete Retention) - 使用简单的分类打标抽取存在哪些事件，看目标中是否保留该集合
    """
    
    DEFAULT_CLAP_MODEL = "laion/clap-htsat-fused"

    def __init__(self, 
                 use_continuous_fidelity: bool = True,
                 use_discrete_matching: bool = True,
                 clap_model_path: Optional[str] = None,
                 device: Optional[str] = None):
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_continuous_fidelity = use_continuous_fidelity
        self.use_discrete_matching = use_discrete_matching
        
        self.clap_model_path = clap_model_path or self.DEFAULT_CLAP_MODEL
        
        self.clap_processor = None
        self.clap_model = None

    def _load_clap_model(self):
        if self.clap_model is not None: return
        print(f"⏳ 正在加载环境声学多模态模型: {self.clap_model_path} ...")
        try:
            from transformers import ClapModel, ClapProcessor
            self.clap_processor = ClapProcessor.from_pretrained(self.clap_model_path)
            self.clap_model = ClapModel.from_pretrained(self.clap_model_path).to(self.device).eval()
            print("✅ CLAP 模型加载成功！此特征提取器对副语言等声音事件高度敏感。")
        except ImportError:
            print("❌ transformers 依赖缺失。请执行: pip install transformers librosa")
            self.use_continuous_fidelity = False
        except Exception as e:
            print(f"❌ CLAP 模型加载失败: {e}")
            self.use_continuous_fidelity = False

    # =============== 特征与事件提取 ===============

    def _extract_clap_embeddings(self, audio_paths: List[str]) -> List[np.ndarray]:
        """提取源和目标的最深层音频空间特征向量"""
        self._load_clap_model()
        if not self.use_continuous_fidelity: return [np.zeros(512)] * len(audio_paths)

        results = []
        for path in tqdm(audio_paths, desc="🔍 计算音频深层声学表征", unit="file"):
            try:
                # CLAP 模型默认采样率常为 48000
                target_sr = 48000 
                audio_array, orig_sr = librosa.load(path, sr=target_sr)
                inputs = self.clap_processor(audio=audio_array, return_tensors="pt", sampling_rate=target_sr)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    # 提取投射过后的跨模态音频嵌入，对于纯音频也能用来做余弦相似计算
                    audio_features = self.clap_model.get_audio_features(**inputs)
                    emb = audio_features[0].cpu().numpy()
                results.append(emb)
            except Exception as e:
                print(f"Failed extracting {path}: {e}")
                results.append(np.zeros(512))
        return results

    def _pseudo_detect_events(self, audio_paths: List[str]) -> List[set]:
        """
        伪事件探测：由于目前开源的高精度 21分类强事件检测还没完全标准化（如WESR API）
        这里放置一个预留桩位。在实际工程中，你可以把这里的逻辑替换为：
        1. 使用 WESR 或 Qwen-Audio 抽取出 [笑声, 咳嗽]
        2. 如果为了跑通 pipeline，这里利用 CLAP 模型，跟预设的文字 prompt 对比给出最可能发生的离散物理现象。
        返回 list of sets, e.g., [{"laugh", "cough"}, set()]
        """
        if not self.use_discrete_matching: return [set()] * len(audio_paths)
        self._load_clap_model()
        
        # 预设要检测的离散非语言事件集合
        candidate_events = ["laughter", "coughing", "sighing", "breathing heavily", "throat clearing"]
        
        results = []
        print("⏳ 离散事件探测 (使用 CLAP 文本对齐 Zero-Shot)...")
        # 预先编码文字候选词
        try:
            text_inputs = self.clap_processor(text=candidate_events, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.clap_model.get_text_features(**text_inputs)
        except Exception:
             return [set()] * len(audio_paths)

        for path in tqdm(audio_paths, desc="🔍 探测物理发声事件", unit="file"):
            try:
                audio_array, _ = librosa.load(path, sr=48000)
                audio_inputs = self.clap_processor(audio=audio_array, return_tensors="pt", sampling_rate=48000)
                audio_inputs = {k: v.to(self.device) for k, v in audio_inputs.items()}
                
                with torch.no_grad():
                    audio_features = self.clap_model.get_audio_features(**audio_inputs)
                    
                # 计算 Cosine similarity between Audio and Text
                audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                similarity = (audio_features @ text_features.T).squeeze(0) 
                
                # 如果相似度过阈值则认为该事件发生（硬切阈值，可依据实际观测降低或升高）
                detected = set()
                thresh = 0.20 # CLAP 默认的置信度域，如果大于表示明确检测到
                sim_scores = similarity.cpu().numpy()
                for idx, score in enumerate(sim_scores):
                    if score > thresh:
                        detected.add(candidate_events[idx])
                results.append(detected)
            except Exception:
                results.append(set())
        
        return results


    # =============== 评测主入口 ===============

    def evaluate_all(self, 
                     source_audio: Union[List[str], str], 
                     target_audio: Union[List[str], str],
                     verbose: bool = True) -> Dict[str, float]:
        """
        全量评测输入：只需源音频与目标生成音频的比对。
        """

        # 加载源路径
        if isinstance(source_audio, str) and os.path.isdir(source_audio):
            src_paths = _load_audio_from_folder(source_audio)
        else:
            src_paths = _load_data_list(source_audio, "Source Audio Paths", "audio")

        # 加载目标路径
        if isinstance(target_audio, str) and os.path.isdir(target_audio):
            tgt_paths = _load_audio_from_folder(target_audio)
        else:
            tgt_paths = _load_data_list(target_audio, "Target Audio Paths", "audio")

        num_samples = len(tgt_paths)
        if num_samples == 0:
            raise ValueError("没有找到目标音频数据，评测停止。")
            
        if len(src_paths) != num_samples:
            raise ValueError(f"数目不一致: Source ({len(src_paths)}) != Target ({num_samples})")

        results = {}

        # ==================== 分支 1：连续特征保真度 (Cosine) ====================
        if self.use_continuous_fidelity:
            if verbose: print(f"\n📝 启动【连续声学表征保真度运算】 ({num_samples} 条)...")
            src_embs = self._extract_clap_embeddings(src_paths)
            tgt_embs = self._extract_clap_embeddings(tgt_paths)

            cosine_sim_total = 0.0 
            valid_count = 0
            
            for i in range(num_samples):
                s_emb = src_embs[i]
                t_emb = tgt_embs[i]
                
                n_s = norm(s_emb)
                n_t = norm(t_emb)
                if n_s > 0 and n_t > 0:
                    sim = np.dot(s_emb, t_emb) / (n_s * n_t)
                    cosine_sim_total += float(sim)
                    valid_count += 1

            final_cosine = (cosine_sim_total / valid_count) if valid_count > 0 else 0.0
            results["Paralinguistic_Fidelity_Cosine"] = round(final_cosine, 4)

        # ==================== 分支 2：离散物理事件保留率 (F1 Set Matching) ====================
        if self.use_discrete_matching:
            if verbose: print(f"\n📝 启动【离散非语言事件匹配/保留率运算】 ({num_samples} 条)...")
            src_events = self._pseudo_detect_events(src_paths)
            tgt_events = self._pseudo_detect_events(tgt_paths)
            
            f1_scores = []
            
            for i in range(num_samples):
                s_set = src_events[i]
                t_set = tgt_events[i]
                
                # 如果源音频确实没有任何副语言事件，那目标就算没生成也是"正常且正确"的 (不拉低也不抬高由于乱发出声音带来的虚假繁荣)
                # 这里如果源没有，且目标也没有，记为完全匹配 1.0；如果目标瞎造了一些出来，记为 0。
                if len(s_set) == 0:
                    score = 1.0 if len(t_set) == 0 else 0.0
                else:
                    # 计算集合 F1
                    intersection = len(s_set.intersection(t_set))
                    precision = intersection / len(t_set) if len(t_set) > 0 else 0.0
                    recall = intersection / len(s_set)
                    if precision + recall > 0:
                        score = 2 * (precision * recall) / (precision + recall)
                    else:
                        score = 0.0
                        
                f1_scores.append(score)
                
            final_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            results["Event_Retention_F1"] = round(final_f1, 4)


        # ==================== 输出反馈 ====================
        if verbose:
            print("\n📊 [ParalinguisticEvaluator] 跨语种副语言综合保留评测报告:")
            print(f"   - 有效评测对齐数: {num_samples}对")
            
            if self.use_continuous_fidelity:
                print("\n   [宏观：连续环境声学质感保真度] (数值范围[-1, 1])")
                print("   该指标判断源和目标的底层喘息、底噪、声学动作基调是否具备同步性：")
                print(f"   => Paralinguistic Fidelity (Cosine): {results.get('Paralinguistic_Fidelity_Cosine', 0):.4f}")
            
            if self.use_discrete_matching:
                print("\n   [微观：显性物理事件存活率] (数值范围[0, 1])")
                print("   打标出源语言中的咳嗽、笑声等事件后，对比目标翻译中是否也强触发了相同动作：")
                print(f"   => Event Retention Rate (F1):        {results.get('Event_Retention_F1', 0):.2%}")
                
        return results
