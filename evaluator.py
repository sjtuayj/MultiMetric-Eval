"""
Model Evaluator - 版本2: 优先本地，无则下载
检查缓存目录是否有模型，有则用本地，无则联网下载
"""
import os
import json
import numpy as np
import sacrebleu
import torch
import whisper
from typing import Dict, List, Optional, Union

try:
    from bleurt import score as bleurt_score
    HAS_BLEURT = True
except ImportError:
    bleurt_score = None
    HAS_BLEURT = False

from comet import download_model, load_from_checkpoint

for var in ["HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"]:
    os.environ.pop(var, None)


class ModelEvaluator:
    """语音翻译评测器 - 混合版（优先本地）"""

    # HuggingFace 缓存目录
    HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

    def __init__(
        self,
        whisper_path: str = "./model/whisper_medium/whisper_models/medium.pt",
        comet_model: str = "Unbabel/wmt22-comet-da",
        bleurt_path: str = "./model/BLEURT-20",
        data_dir: str = "./internal_data",
        dataset_file: str = "dataset_paired.json",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 初始化评测器 (设备: {self.device})")

        self.dataset = self._load_dataset(data_dir, dataset_file)
        self.asr_model = self._load_whisper(whisper_path)
        self.comet_model = self._load_comet(comet_model)
        self.bleurt_scorer = self._load_bleurt(bleurt_path) if HAS_BLEURT else None

        print("✅ 系统就绪！")

    def _load_dataset(self, data_dir: str, filename: str) -> Dict[str, List[str]]:
        path = os.path.join(data_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            pairs = json.load(f)

        dataset = {"source_speech": [], "source_text": [], "reference_text": []}
        for p in pairs:
            speech_path = p["source_speech_path"]
            if not os.path.exists(speech_path):
                alt = os.path.join(data_dir, "audio", os.path.basename(speech_path))
                if os.path.exists(alt):
                    speech_path = alt
            dataset["source_speech"].append(speech_path)
            dataset["source_text"].append(p["source_text"])
            dataset["reference_text"].append(p["reference_text"])

        print(f"✅ 数据集加载完毕 ({len(dataset['source_speech'])} 条)")
        return dataset

    def _is_cached(self, model_name: str) -> bool:
        """检查 HuggingFace 缓存是否有该模型"""
        cache_dir = os.path.join(self.HF_CACHE, f"models--{model_name.replace('/', '--')}")
        return os.path.exists(cache_dir)

    def _load_whisper(self, path: str):
        if os.path.exists(path):
            print(f"⏳ [Local] 加载 Whisper: {path}")
        else:
            print(f"⏳ [Online] 下载 Whisper: {path}")
        return whisper.load_model(path, device=self.device)

    def _load_comet(self, model_name: str):
        if self._is_cached(model_name):
            print(f"⏳ [Cache] 使用缓存的 COMET: {model_name}")
        else:
            print(f"⏳ [Online] 下载 COMET: {model_name}")
        
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        print("✅ COMET 加载成功！")
        return model.to(self.device) if self.device == "cuda" else model

    def _load_bleurt(self, path: str):
        if not os.path.exists(path):
            print("⚠️ BLEURT 未找到")
            return None
        print("⏳ 加载 BLEURT...")
        try:
            return bleurt_score.BleurtScorer(path)
        except Exception as e:
            print(f"⚠️ BLEURT 加载失败: {e}")
            return None

    def _transcribe(self, audio_paths: List[str]) -> List[str]:
        print("🎤 ASR 转写中...")
        results = []
        for path in audio_paths:
            if not os.path.exists(path):
                results.append("")
                continue
            try:
                text = self.asr_model.transcribe(path, fp16=(self.device == "cuda"))
                results.append(text["text"].strip())
            except:
                results.append("")
        return results

    def _calc_bleu(self, hyps: List[str], refs: List[str]) -> float:
        try:
            return round(sacrebleu.corpus_bleu(hyps, [refs]).score, 4)
        except:
            return 0.0

    def _calc_chrf(self, hyps: List[str], refs: List[str]) -> float:
        try:
            return round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 4)
        except:
            return 0.0

    def _calc_bleurt(self, hyps: List[str], refs: List[str]) -> float:
        if not self.bleurt_scorer:
            return -1.0
        try:
            scores = self.bleurt_scorer.score(references=refs, candidates=hyps)
            return round(float(np.mean(scores)), 4)
        except:
            return -1.0

    def _calc_comet(self, srcs: List[str], hyps: List[str], refs: List[str]) -> float:
        if not self.comet_model:
            return -1.0
        try:
            data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
            gpus = 1 if self.device == "cuda" else 0
            result = self.comet_model.predict(data, batch_size=8, gpus=gpus)
            return round(result.system_score, 4)
        except:
            return -1.0

    def get_test_source(self) -> Dict[str, List[str]]:
        return {
            "source_speech": self.dataset["source_speech"],
            "source_text": self.dataset["source_text"],
        }

    def evaluate(
        self,
        use_internal_dataset: bool,
        user_data: Dict[str, Optional[List[str]]],
    ) -> Dict[str, Union[float, List[str]]]:
        
        if use_internal_dataset:
            refs = self.dataset["reference_text"]
            srcs = self.dataset["source_text"]
        else:
            refs = user_data["reference_text"]
            srcs = user_data["source_text"]

        target_text = user_data.get("target_text")
        if not target_text or all(t is None for t in target_text):
            hyps = self._transcribe(user_data.get("target_speech", []))
        else:
            hyps = target_text

        print("📊 计算指标...")
        return {
            "sacreBLEU": self._calc_bleu(hyps, refs),
            "chrF++": self._calc_chrf(hyps, refs),
            "BLEURT": self._calc_bleurt(hyps, refs),
            "COMET": self._calc_comet(srcs, hyps, refs),
            "final_target_text": hyps,
        }


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    test_data = evaluator.get_test_source()
    dummy = {"target_text": ["Test"] * len(test_data["source_text"])}
    results = evaluator.evaluate(True, dummy)

    print("\n📈 评测结果:")
    for k, v in results.items():
        if k != "final_target_text":
            print(f"   {k}: {v}")
