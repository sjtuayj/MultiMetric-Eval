import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
from numpy.linalg import norm
from tqdm import tqdm


@dataclass(frozen=True)
class EventSpan:
    label: str
    start_ms: int
    end_ms: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "label": self.label,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }


@dataclass(frozen=True)
class ParalinguisticSample:
    sample_id: str
    source_audio: str
    source_text: str = ""
    events: List[EventSpan] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_event_annotation(self) -> List[Dict[str, int]]:
        return [event.to_dict() for event in self.events]


def ensure_existing_audio(path: str) -> str:
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {path}")
    return str(audio_path.resolve())


def load_paralinguistic_manifest(path: str) -> List[ParalinguisticSample]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Paralinguistic manifest not found: {path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Paralinguistic manifest must be a JSON list.")

    samples: List[ParalinguisticSample] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        source_audio = item.get("source_audio")
        if not source_audio:
            continue

        raw_events = item.get("events", [])
        if not isinstance(raw_events, list):
            raise ValueError(f"Manifest item {index} has invalid 'events' field.")

        events: List[EventSpan] = []
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                continue
            label = raw_event.get("label")
            start_ms = raw_event.get("start_ms")
            end_ms = raw_event.get("end_ms")
            if label is None or start_ms is None or end_ms is None:
                continue
            try:
                start_ms = int(start_ms)
                end_ms = int(end_ms)
            except Exception:
                continue
            if end_ms <= start_ms:
                continue
            events.append(EventSpan(label=str(label), start_ms=start_ms, end_ms=end_ms))

        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        samples.append(
            ParalinguisticSample(
                sample_id=str(item.get("id", index)),
                source_audio=ensure_existing_audio(str(source_audio)),
                source_text=str(item.get("source_text", "")).strip(),
                events=events,
                metadata=metadata,
            )
        )

    if not samples:
        raise RuntimeError(f"No usable paralinguistic samples found in {path}")

    return samples


def load_paralinguistic_samples(path: str, max_samples: Optional[int] = None) -> List[ParalinguisticSample]:
    samples = load_paralinguistic_manifest(path)
    if max_samples is not None:
        return samples[:max_samples]
    return samples


def build_paralinguistic_inputs(samples: List[ParalinguisticSample]) -> Dict[str, List[Any]]:
    if not samples:
        raise ValueError("No paralinguistic samples provided.")
    return {
        "sample_ids": [sample.sample_id for sample in samples],
        "source_audio": [sample.source_audio for sample in samples],
        "source_text": [sample.source_text for sample in samples],
        "metadata": [sample.metadata for sample in samples],
    }


def _load_data_list(input_data: Union[str, List[str]], name: str) -> List[str]:
    if isinstance(input_data, list):
        return input_data
    if not isinstance(input_data, str):
        raise ValueError(f"{name} must be a path or a list of paths.")

    path = Path(input_data)
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {input_data}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{name} JSON must be a list.")
        return data

    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_audio_from_folder(folder_path: str, extensions: Tuple[str, ...] = (".wav", ".mp3", ".flac")) -> List[str]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    audio_files = []
    for ext in extensions:
        audio_files.extend(folder.glob(f"*{ext}"))

    audio_files = sorted(audio_files, key=lambda x: x.stem)
    if not audio_files:
        raise ValueError(f"No audio files found under: {folder_path}")

    return [str(p) for p in audio_files]


class ParalinguisticEvaluator:
    DEFAULT_CLAP_MODEL = "laion/clap-htsat-fused"

    def __init__(
        self,
        use_continuous_fidelity: bool = True,
        clap_model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_continuous_fidelity = use_continuous_fidelity
        self.clap_model_path = clap_model_path or self.DEFAULT_CLAP_MODEL

        self.clap_processor = None
        self.clap_model = None

    def _load_clap_model(self):
        if self.clap_model is not None:
            return

        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise RuntimeError("ParalinguisticEvaluator requires transformers to load CLAP.") from exc

        self.clap_processor = ClapProcessor.from_pretrained(self.clap_model_path)
        self.clap_model = ClapModel.from_pretrained(self.clap_model_path).to(self.device).eval()

    def _extract_clap_embeddings(self, audio_paths: List[str]) -> List[np.ndarray]:
        self._load_clap_model()

        results = []
        for path in tqdm(audio_paths, desc="Extracting CLAP embeddings", unit="file"):
            audio_array, sr = librosa.load(path, sr=48000)
            inputs = self.clap_processor(audio=audio_array, return_tensors="pt", sampling_rate=sr)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                audio_features = self.clap_model.get_audio_features(**inputs)
            results.append(audio_features[0].detach().cpu().numpy())

        return results

    def evaluate_all(
        self,
        source_audio: Union[List[str], str],
        target_audio: Union[List[str], str],
        verbose: bool = True,
        return_diagnostics: bool = False,
    ):
        if isinstance(source_audio, str) and os.path.isdir(source_audio):
            src_paths = _load_audio_from_folder(source_audio)
        else:
            src_paths = _load_data_list(source_audio, "Source Audio")

        if isinstance(target_audio, str) and os.path.isdir(target_audio):
            tgt_paths = _load_audio_from_folder(target_audio)
        else:
            tgt_paths = _load_data_list(target_audio, "Target Audio")

        if len(src_paths) != len(tgt_paths):
            raise ValueError(f"Source and target size mismatch: {len(src_paths)} vs {len(tgt_paths)}")

        num_samples = len(src_paths)
        if num_samples == 0:
            raise ValueError("No samples found for paralinguistic evaluation.")

        results: Dict[str, float] = {}
        diagnostics: Dict[str, Any] = {"num_samples": num_samples}

        if self.use_continuous_fidelity:
            src_embs = self._extract_clap_embeddings(src_paths)
            tgt_embs = self._extract_clap_embeddings(tgt_paths)

            cosine_total = 0.0
            valid_count = 0
            for src_emb, tgt_emb in zip(src_embs, tgt_embs):
                n_src = norm(src_emb)
                n_tgt = norm(tgt_emb)
                if n_src > 0 and n_tgt > 0:
                    cosine_total += float(np.dot(src_emb, tgt_emb) / (n_src * n_tgt))
                    valid_count += 1

            results["Paralinguistic_Fidelity_Cosine"] = round(cosine_total / valid_count, 4) if valid_count > 0 else 0.0

        if verbose:
            print("\n[ParalinguisticEvaluator] Summary")
            print(f"  Samples: {num_samples}")
            for key, value in results.items():
                print(f"  {key}: {value}")

        if return_diagnostics:
            return results, diagnostics
        return results


def evaluate_paralinguistic_dataset(
    target_audio: List[str],
    *,
    samples: Optional[List[ParalinguisticSample]] = None,
    manifest_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    evaluator: Optional[ParalinguisticEvaluator] = None,
    evaluator_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    return_diagnostics: bool = False,
    sample_transform: Optional[Callable[[List[ParalinguisticSample]], List[ParalinguisticSample]]] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]] | Dict[str, float]:
    if samples is None:
        if not manifest_path:
            raise ValueError("manifest_path is required when samples are not provided.")
        samples = load_paralinguistic_manifest(manifest_path)

    if max_samples is not None:
        samples = samples[:max_samples]

    if sample_transform is not None:
        samples = sample_transform(samples)

    inputs = build_paralinguistic_inputs(samples)
    if len(inputs["source_audio"]) != len(target_audio):
        raise ValueError(
            f"Source/target size mismatch for paralinguistic evaluation: "
            f"{len(inputs['source_audio'])} vs {len(target_audio)}"
        )

    if evaluator is None:
        final_kwargs = dict(evaluator_kwargs or {})
        evaluator = ParalinguisticEvaluator(
            use_continuous_fidelity=True,
            device=device,
            **final_kwargs,
        )

    result = evaluator.evaluate_all(
        source_audio=inputs["source_audio"],
        target_audio=target_audio,
        verbose=True,
        return_diagnostics=return_diagnostics,
    )

    if return_diagnostics:
        scores, diagnostics = result
        diagnostics["sample_ids"] = inputs["sample_ids"]
        diagnostics["metadata"] = inputs["metadata"]
        return scores, diagnostics

    return result
