import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


LabelNormalizer = Optional[Union[Dict[str, Optional[str]], Callable[[str], Optional[str]]]]


@dataclass(frozen=True)
class ParalinguisticSample:
    sample_id: str
    source_audio: str
    source_text: str = ""
    source_label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EventPrediction:
    label: Optional[str]
    score: Optional[float] = None
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "score": None if self.score is None else float(self.score),
            "scores": {str(key): float(value) for key, value in self.scores.items()},
        }


@dataclass(frozen=True)
class EventPredictionConfig:
    score_threshold: float = 0.2
    fallback_top1: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_threshold": float(self.score_threshold),
            "fallback_top1": bool(self.fallback_top1),
        }


class BaseAudioEventPredictor(ABC):
    @abstractmethod
    def predict(
        self,
        audio_paths: Sequence[str],
        candidate_labels: Sequence[str],
    ) -> List[EventPrediction]:
        raise NotImplementedError


def ensure_existing_audio(path: str) -> str:
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {path}")
    return str(audio_path.resolve())


def _to_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_audio_mono(audio_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if target_sr is not None and sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    waveform = waveform.squeeze(0).contiguous().float().cpu().numpy()
    return waveform, int(sr)


def _normalize_text_label(label: str) -> str:
    return " ".join(str(label).strip().split())


def _apply_label_normalizer(label: Optional[str], label_normalizer: LabelNormalizer) -> Optional[str]:
    if label is None:
        return None

    normalized = _normalize_text_label(label)
    if not normalized:
        return None

    if label_normalizer is None:
        mapped = normalized
    elif callable(label_normalizer):
        mapped = label_normalizer(normalized)
    else:
        mapped = label_normalizer.get(normalized, normalized)

    if mapped is None:
        return None
    mapped_text = _normalize_text_label(str(mapped))
    return mapped_text or None


def _coerce_manifest_source_label(raw_label: Any, index: int) -> Optional[str]:
    if raw_label is None:
        return None

    if isinstance(raw_label, list):
        cleaned = []
        for item in raw_label:
            label = _normalize_text_label(str(item))
            if label and label not in cleaned:
                cleaned.append(label)
        if not cleaned:
            return None
        if len(cleaned) > 1:
            raise ValueError(
                f"Manifest item {index} has multiple source labels {cleaned}. "
                "This evaluator expects at most one source event label per sample."
            )
        return cleaned[0]

    label = _normalize_text_label(str(raw_label))
    return label or None


def _normalize_label_batch(
    labels: Optional[Sequence[Optional[str]]],
    *,
    name: str,
    expected_length: int,
    label_normalizer: LabelNormalizer,
) -> Optional[List[Optional[str]]]:
    if labels is None:
        return None
    if len(labels) != expected_length:
        raise ValueError(f"{name} size mismatch: {len(labels)} vs {expected_length}")

    normalized: List[Optional[str]] = []
    for index, label in enumerate(labels):
        if label is None:
            normalized.append(None)
            continue
        if not isinstance(label, str):
            raise ValueError(f"{name}[{index}] must be a string or None.")
        normalized.append(_apply_label_normalizer(label, label_normalizer))
    return normalized


def _normalize_candidate_labels(
    candidate_labels: Optional[Sequence[str]],
    *,
    label_normalizer: LabelNormalizer,
) -> List[str]:
    if candidate_labels is None:
        return []

    normalized: List[str] = []
    seen = set()
    for label in candidate_labels:
        mapped = _apply_label_normalizer(label, label_normalizer)
        if mapped is None or mapped in seen:
            continue
        normalized.append(mapped)
        seen.add(mapped)
    return normalized


def _resolve_candidate_labels(
    *,
    candidate_labels: Optional[Sequence[str]],
    source_labels: Optional[Sequence[Optional[str]]],
    target_labels: Optional[Sequence[Optional[str]]],
    label_normalizer: LabelNormalizer,
) -> List[str]:
    resolved = _normalize_candidate_labels(candidate_labels, label_normalizer=label_normalizer)
    if resolved:
        return resolved

    seen = set()
    derived: List[str] = []
    for batch in (source_labels, target_labels):
        if batch is None:
            continue
        for label in batch:
            mapped = _apply_label_normalizer(label, label_normalizer)
            if mapped is None or mapped in seen:
                continue
            derived.append(mapped)
            seen.add(mapped)
    return derived


def _load_data_list(data: Union[List[str], str], name: str) -> List[str]:
    if isinstance(data, str):
        path = Path(data)
        if path.exists() and path.is_dir():
            return load_audio_from_folder(str(path))
        return [ensure_existing_audio(data)]
    if not isinstance(data, list):
        raise ValueError(f"{name} must be a path or a list of paths.")
    return [ensure_existing_audio(str(item)) for item in data]


def load_audio_from_folder(folder_path: str) -> List[str]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not folder.is_dir():
        raise ValueError(f"Expected a directory, got: {folder_path}")

    audio_files: List[Path] = []
    for extension in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
        audio_files.extend(folder.glob(f"*{extension}"))
    audio_files = sorted(audio_files, key=lambda item: item.stem)
    if not audio_files:
        raise ValueError(f"No audio files found under: {folder_path}")
    return [str(item.resolve()) for item in audio_files]


def _safe_mean(values: Sequence[float]) -> float:
    return round(float(sum(values) / len(values)), 4) if values else 0.0


def _compute_single_label_metrics(
    reference_labels: Sequence[Optional[str]],
    predicted_labels: Sequence[Optional[str]],
    *,
    class_labels: Sequence[str],
) -> Dict[str, Any]:
    evaluated_indices = [index for index, label in enumerate(reference_labels) if label is not None]
    evaluated_reference = [reference_labels[index] for index in evaluated_indices]
    evaluated_prediction = [predicted_labels[index] for index in evaluated_indices]

    total = len(evaluated_reference)
    correct = sum(
        1
        for reference_label, predicted_label in zip(evaluated_reference, evaluated_prediction)
        if reference_label == predicted_label
    )
    abstained = sum(1 for predicted_label in evaluated_prediction if predicted_label is None)

    unique_class_labels = list(dict.fromkeys([label for label in class_labels if label]))
    if not unique_class_labels:
        unique_class_labels = sorted(
            {
                label
                for label in evaluated_reference + evaluated_prediction
                if label is not None
            }
        )

    per_label: Dict[str, Dict[str, float]] = {}
    macro_f1_values: List[float] = []
    macro_recall_values: List[float] = []
    confusion: Dict[str, Dict[str, int]] = {}

    for reference_label, predicted_label in zip(evaluated_reference, evaluated_prediction):
        predicted_key = predicted_label if predicted_label is not None else "__none__"
        confusion.setdefault(str(reference_label), {})
        confusion[str(reference_label)][predicted_key] = confusion[str(reference_label)].get(predicted_key, 0) + 1

    for label in unique_class_labels:
        tp = sum(
            1
            for reference_label, predicted_label in zip(evaluated_reference, evaluated_prediction)
            if reference_label == label and predicted_label == label
        )
        fp = sum(
            1
            for reference_label, predicted_label in zip(evaluated_reference, evaluated_prediction)
            if reference_label != label and predicted_label == label
        )
        fn = sum(
            1
            for reference_label, predicted_label in zip(evaluated_reference, evaluated_prediction)
            if reference_label == label and predicted_label != label
        )
        support = sum(1 for reference_label in evaluated_reference if reference_label == label)

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

        per_label[label] = {
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        macro_f1_values.append(f1)
        macro_recall_values.append(recall)

    return {
        "num_evaluated": int(total),
        "num_skipped": int(len(reference_labels) - total),
        "num_correct": int(correct),
        "num_abstained": int(abstained),
        "preservation_rate": round(float(correct / total), 4) if total > 0 else 0.0,
        "macro_f1": _safe_mean(macro_f1_values),
        "macro_recall": _safe_mean(macro_recall_values),
        "per_label": per_label,
        "confusion_matrix": confusion,
    }


class ClapAudioEventPredictor(BaseAudioEventPredictor):
    PROMPT_TEMPLATES = (
        "{label}",
        "the sound of {label}",
        "an audio recording of {label}",
        "a person producing {label}",
    )

    def __init__(
        self,
        *,
        model_path: str = "laion/clap-htsat-fused",
        config: Optional[EventPredictionConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.config = config or EventPredictionConfig()
        self.device = _to_device(device)
        self._processor = None
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise RuntimeError("CLAP-based event prediction requires `transformers`.") from exc

        self._processor = ClapProcessor.from_pretrained(self.model_path)
        self._model = ClapModel.from_pretrained(self.model_path).to(self.device).eval()

    def _extract_audio_embeddings(self, audio_paths: Sequence[str]) -> List[np.ndarray]:
        self._load_model()
        assert self._processor is not None
        assert self._model is not None

        embeddings: List[np.ndarray] = []
        for audio_path in tqdm(audio_paths, desc="Extracting CLAP event embeddings", unit="file"):
            waveform, sr = _load_audio_mono(str(audio_path), target_sr=48000)
            inputs = self._processor(audio=waveform, sampling_rate=sr, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                features = self._model.get_audio_features(**inputs)
            embeddings.append(features[0].detach().cpu().numpy())
        return embeddings

    def _extract_text_embeddings(self, texts: Sequence[str]) -> List[np.ndarray]:
        self._load_model()
        assert self._processor is not None
        assert self._model is not None

        embeddings: List[np.ndarray] = []
        for text in texts:
            inputs = self._processor(text=[str(text)], return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                features = self._model.get_text_features(**inputs)
            embeddings.append(features[0].detach().cpu().numpy())
        return embeddings

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(embedding))
        if norm <= 0.0:
            return embedding
        return embedding / norm

    def _build_prompts(self, candidate_labels: Sequence[str]) -> List[Tuple[str, str]]:
        prompt_records: List[Tuple[str, str]] = []
        for label in candidate_labels:
            for template in self.PROMPT_TEMPLATES:
                prompt_records.append((label, template.format(label=label)))
        return prompt_records

    def predict(
        self,
        audio_paths: Sequence[str],
        candidate_labels: Sequence[str],
    ) -> List[EventPrediction]:
        normalized_candidate_labels = [
            _normalize_text_label(label)
            for label in candidate_labels
            if _normalize_text_label(label)
        ]
        unique_candidate_labels = list(dict.fromkeys(normalized_candidate_labels))
        if not unique_candidate_labels:
            return [EventPrediction(label=None, score=None, scores={}) for _ in audio_paths]

        prompt_records = self._build_prompts(unique_candidate_labels)
        audio_embeddings = [self._normalize_embedding(item) for item in self._extract_audio_embeddings(audio_paths)]
        text_embeddings = [
            self._normalize_embedding(item)
            for item in self._extract_text_embeddings([prompt for _, prompt in prompt_records])
        ]

        predictions: List[EventPrediction] = []
        threshold = float(self.config.score_threshold)
        fallback_top1 = bool(self.config.fallback_top1)

        for audio_embedding in audio_embeddings:
            label_scores: Dict[str, float] = {}
            for (label, _prompt), text_embedding in zip(prompt_records, text_embeddings):
                score = float(np.dot(audio_embedding, text_embedding))
                if label not in label_scores or score > label_scores[label]:
                    label_scores[label] = score

            top_label = max(label_scores.items(), key=lambda item: item[1])[0]
            top_score = float(label_scores[top_label])

            predicted_label: Optional[str]
            if top_score >= threshold or fallback_top1:
                predicted_label = top_label
            else:
                predicted_label = None

            predictions.append(
                EventPrediction(
                    label=predicted_label,
                    score=top_score,
                    scores={label: round(score, 4) for label, score in sorted(label_scores.items())},
                )
            )

        return predictions


class ParalinguisticEvaluator:
    DEFAULT_CLAP_MODEL = "laion/clap-htsat-fused"

    def __init__(
        self,
        use_continuous_fidelity: bool = True,
        use_event_preservation: bool = True,
        clap_model_path: Optional[str] = None,
        event_predictor: Optional[BaseAudioEventPredictor] = None,
        event_prediction_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        **_: Any,
    ) -> None:
        self.device = _to_device(device)
        self.use_continuous_fidelity = bool(use_continuous_fidelity)
        self.use_event_preservation = bool(use_event_preservation)
        self.clap_model_path = clap_model_path or self.DEFAULT_CLAP_MODEL
        self.event_prediction_config = EventPredictionConfig(**(event_prediction_config or {}))
        self.event_predictor = event_predictor

        self._clap_processor = None
        self._clap_model = None
        self._default_predictor: Optional[ClapAudioEventPredictor] = None

    def _load_clap(self) -> None:
        if self._clap_model is not None and self._clap_processor is not None:
            return

        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise RuntimeError("Paralinguistic_Fidelity_Cosine requires `transformers` to load CLAP.") from exc

        self._clap_processor = ClapProcessor.from_pretrained(self.clap_model_path)
        self._clap_model = ClapModel.from_pretrained(self.clap_model_path).to(self.device).eval()

    def _extract_clap_embeddings(self, audio_paths: Sequence[str]) -> List[np.ndarray]:
        self._load_clap()
        assert self._clap_processor is not None
        assert self._clap_model is not None

        embeddings: List[np.ndarray] = []
        for audio_path in tqdm(audio_paths, desc="Extracting CLAP embeddings", unit="file"):
            waveform, sr = _load_audio_mono(str(audio_path), target_sr=48000)
            inputs = self._clap_processor(audio=waveform, sampling_rate=sr, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                features = self._clap_model.get_audio_features(**inputs)
            embeddings.append(features[0].detach().cpu().numpy())
        return embeddings

    @staticmethod
    def _average_cosine(source_embeddings: Sequence[np.ndarray], target_embeddings: Sequence[np.ndarray]) -> float:
        total = 0.0
        count = 0
        for source_embedding, target_embedding in zip(source_embeddings, target_embeddings):
            source_norm = float(np.linalg.norm(source_embedding))
            target_norm = float(np.linalg.norm(target_embedding))
            if source_norm <= 0.0 or target_norm <= 0.0:
                continue
            total += float(np.dot(source_embedding, target_embedding) / (source_norm * target_norm))
            count += 1
        return round(total / count, 4) if count > 0 else 0.0

    def _get_event_predictor(self) -> BaseAudioEventPredictor:
        if self.event_predictor is not None:
            return self.event_predictor
        if self._default_predictor is None:
            self._default_predictor = ClapAudioEventPredictor(
                model_path=self.clap_model_path,
                config=self.event_prediction_config,
                device=self.device,
            )
        return self._default_predictor

    def _predict_labels(
        self,
        *,
        audio_paths: Sequence[str],
        candidate_labels: Sequence[str],
        label_normalizer: LabelNormalizer,
    ) -> List[EventPrediction]:
        predictor = self._get_event_predictor()
        if not hasattr(predictor, "predict"):
            raise TypeError("event_predictor must expose a `predict(audio_paths, candidate_labels)` method.")

        predictions = predictor.predict(audio_paths, candidate_labels)
        if len(predictions) != len(audio_paths):
            raise ValueError(
                "event_predictor returned a different number of predictions than audio inputs: "
                f"{len(predictions)} vs {len(audio_paths)}"
            )

        normalized_predictions: List[EventPrediction] = []
        for prediction in predictions:
            normalized_predictions.append(
                EventPrediction(
                    label=_apply_label_normalizer(prediction.label, label_normalizer),
                    score=prediction.score,
                    scores={
                        _apply_label_normalizer(label, label_normalizer) or label: float(score)
                        for label, score in prediction.scores.items()
                    },
                )
            )
        return normalized_predictions

    def evaluate_all(
        self,
        source_audio: Union[List[str], str],
        target_audio: Union[List[str], str],
        *,
        source_labels: Optional[Sequence[Optional[str]]] = None,
        target_labels: Optional[Sequence[Optional[str]]] = None,
        candidate_labels: Optional[Sequence[str]] = None,
        label_normalizer: LabelNormalizer = None,
        sample_ids: Optional[Sequence[str]] = None,
        verbose: bool = True,
        return_diagnostics: bool = False,
    ) -> Union[Tuple[Dict[str, float], Dict[str, Any]], Dict[str, float]]:
        source_audio_paths = _load_data_list(source_audio, "Source Audio")
        target_audio_paths = _load_data_list(target_audio, "Target Audio")

        if len(source_audio_paths) != len(target_audio_paths):
            raise ValueError(
                f"Source and target size mismatch: {len(source_audio_paths)} vs {len(target_audio_paths)}"
            )
        if not source_audio_paths:
            raise ValueError("No samples found for paralinguistic evaluation.")

        num_samples = len(source_audio_paths)
        normalized_source_labels = _normalize_label_batch(
            source_labels,
            name="source_labels",
            expected_length=num_samples,
            label_normalizer=label_normalizer,
        )
        normalized_target_labels = _normalize_label_batch(
            target_labels,
            name="target_labels",
            expected_length=num_samples,
            label_normalizer=label_normalizer,
        )
        resolved_candidate_labels = _resolve_candidate_labels(
            candidate_labels=candidate_labels,
            source_labels=normalized_source_labels,
            target_labels=normalized_target_labels,
            label_normalizer=label_normalizer,
        )

        results: Dict[str, float] = {}
        diagnostics: Dict[str, Any] = {
            "num_samples": num_samples,
            "device": self.device,
            "clap_model_path": self.clap_model_path,
        }

        if self.use_continuous_fidelity:
            source_embeddings = self._extract_clap_embeddings(source_audio_paths)
            target_embeddings = self._extract_clap_embeddings(target_audio_paths)
            cosine = self._average_cosine(source_embeddings, target_embeddings)
            results["Paralinguistic_Fidelity_Cosine"] = cosine
            diagnostics["continuous_fidelity"] = {
                "metric": "Paralinguistic_Fidelity_Cosine",
                "num_embeddings": len(source_embeddings),
                "score": cosine,
            }

        if self.use_event_preservation:
            if (normalized_source_labels is None or normalized_target_labels is None) and not resolved_candidate_labels:
                raise ValueError(
                    "Event preservation requires candidate_labels when either source_labels or "
                    "target_labels are not provided."
                )

            source_prediction_records: Optional[List[EventPrediction]] = None
            target_prediction_records: Optional[List[EventPrediction]] = None
            source_label_origin = "provided_source_labels"
            target_label_origin = "provided_target_labels"

            if normalized_source_labels is None:
                source_prediction_records = self._predict_labels(
                    audio_paths=source_audio_paths,
                    candidate_labels=resolved_candidate_labels,
                    label_normalizer=label_normalizer,
                )
                normalized_source_labels = [prediction.label for prediction in source_prediction_records]
                source_label_origin = "predicted_source_labels"

            if normalized_target_labels is None:
                target_prediction_records = self._predict_labels(
                    audio_paths=target_audio_paths,
                    candidate_labels=resolved_candidate_labels,
                    label_normalizer=label_normalizer,
                )
                normalized_target_labels = [prediction.label for prediction in target_prediction_records]
                target_label_origin = "predicted_target_labels"

            assert normalized_source_labels is not None
            assert normalized_target_labels is not None

            metric_payload = _compute_single_label_metrics(
                normalized_source_labels,
                normalized_target_labels,
                class_labels=resolved_candidate_labels,
            )

            use_predicted_reference = source_labels is None
            if use_predicted_reference:
                rate_name = "Predicted_Event_Consistency_Rate"
                macro_f1_name = "Predicted_Event_Consistency_Macro_F1"
                macro_recall_name = "Predicted_Event_Consistency_Macro_Recall"
            else:
                rate_name = "Acoustic_Event_Preservation_Rate"
                macro_f1_name = "Acoustic_Event_Preservation_Macro_F1"
                macro_recall_name = "Acoustic_Event_Preservation_Macro_Recall"

            results[rate_name] = metric_payload["preservation_rate"]
            results[macro_f1_name] = metric_payload["macro_f1"]
            results[macro_recall_name] = metric_payload["macro_recall"]

            diagnostics["event_preservation"] = {
                "candidate_labels": list(resolved_candidate_labels),
                "source_label_origin": source_label_origin,
                "target_label_origin": target_label_origin,
                "config": self.event_prediction_config.to_dict(),
                "num_evaluated": metric_payload["num_evaluated"],
                "num_skipped": metric_payload["num_skipped"],
                "num_abstained": metric_payload["num_abstained"],
                "per_label": metric_payload["per_label"],
                "confusion_matrix": metric_payload["confusion_matrix"],
                "source_predictions": [prediction.to_dict() for prediction in source_prediction_records]
                if source_prediction_records is not None
                else None,
                "target_predictions": [prediction.to_dict() for prediction in target_prediction_records]
                if target_prediction_records is not None
                else None,
                "samples": [
                    {
                        "sample_index": index,
                        "sample_id": sample_ids[index] if sample_ids is not None else str(index),
                        "reference_label": normalized_source_labels[index],
                        "predicted_label": normalized_target_labels[index],
                        "correct": normalized_source_labels[index] is not None
                        and normalized_source_labels[index] == normalized_target_labels[index],
                    }
                    for index in range(num_samples)
                ],
            }

        if verbose:
            print("\n[ParalinguisticEvaluator] Summary")
            print(f"  Samples: {num_samples}")
            for metric_name, score in results.items():
                print(f"  {metric_name}: {score}")

        if return_diagnostics:
            return results, diagnostics
        return results


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
            raise ValueError(f"Manifest item {index} must be a dict.")

        source_audio = item.get("source_audio")
        if not source_audio:
            raise ValueError(f"Manifest item {index} is missing source_audio.")

        raw_label = item.get("source_label", item.get("label", item.get("labels")))
        source_label = _coerce_manifest_source_label(raw_label, index)
        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        samples.append(
            ParalinguisticSample(
                sample_id=str(item.get("id", index)),
                source_audio=ensure_existing_audio(str(source_audio)),
                source_text=str(item.get("source_text", "")).strip(),
                source_label=source_label,
                metadata=metadata,
            )
        )
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
        "source_labels": [sample.source_label for sample in samples],
        "metadata": [sample.metadata for sample in samples],
    }


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
    target_labels: Optional[Sequence[Optional[str]]] = None,
    candidate_labels: Optional[Sequence[str]] = None,
    label_normalizer: LabelNormalizer = None,
) -> Union[Tuple[Dict[str, float], Dict[str, Any]], Dict[str, float]]:
    if samples is None:
        if not manifest_path:
            raise ValueError("manifest_path is required when samples are not provided.")
        samples = load_paralinguistic_manifest(manifest_path)

    if max_samples is not None:
        samples = samples[:max_samples]
    if sample_transform is not None:
        samples = sample_transform(samples)
    if not samples:
        raise ValueError("No paralinguistic samples available.")

    inputs = build_paralinguistic_inputs(samples)
    if len(inputs["source_audio"]) != len(target_audio):
        raise ValueError(
            f"Source/target size mismatch for paralinguistic evaluation: "
            f"{len(inputs['source_audio'])} vs {len(target_audio)}"
        )

    if evaluator is None:
        final_kwargs = dict(evaluator_kwargs or {})
        final_kwargs.setdefault("use_continuous_fidelity", True)
        final_kwargs.setdefault("use_event_preservation", True)
        evaluator = ParalinguisticEvaluator(device=device, **final_kwargs)

    result = evaluator.evaluate_all(
        source_audio=inputs["source_audio"],
        target_audio=target_audio,
        source_labels=inputs["source_labels"],
        target_labels=target_labels,
        candidate_labels=candidate_labels,
        label_normalizer=label_normalizer,
        sample_ids=inputs["sample_ids"],
        verbose=True,
        return_diagnostics=return_diagnostics,
    )

    if return_diagnostics:
        scores, diagnostics = result
        diagnostics["sample_ids"] = inputs["sample_ids"]
        diagnostics["metadata"] = inputs["metadata"]
        return scores, diagnostics

    return result
