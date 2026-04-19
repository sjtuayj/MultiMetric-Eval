import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


EventLike = Union["EventSpan", Dict[str, Any], str]
LabelLike = Union[str, Dict[str, Any], "EventSpan"]
EventLabelMapper = Optional[Union[Dict[str, Optional[str]], Callable[[str], Optional[str]]]]


@dataclass(frozen=True)
class EventSpan:
    label: str
    start_ms: int
    end_ms: int
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "label": self.label,
            "start_ms": int(self.start_ms),
            "end_ms": int(self.end_ms),
        }
        if self.score is not None:
            payload["score"] = float(self.score)
        return payload


@dataclass(frozen=True)
class ParalinguisticSample:
    sample_id: str
    source_audio: str
    source_text: str = ""
    events: List[EventSpan] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_event_annotation(self) -> List[Dict[str, Any]]:
        return [event.to_dict() for event in self.events]


@dataclass(frozen=True)
class DiscreteEventConfig:
    enabled: bool = False
    beats_model_path: Optional[str] = None
    detector_model_path: Optional[str] = None
    score_threshold: float = 0.05
    clap_label_score_threshold: float = 0.2
    clap_label_fallback_top1: bool = False
    window_size_s: float = 1.0
    hop_size_s: float = 0.25
    inference_batch_size: int = 8
    min_event_duration_ms: int = 120
    merge_gap_ms: int = 150
    onset_tolerance_ms: int = 200
    offset_tolerance_ms: int = 200
    offset_tolerance_ratio: float = 0.2
    allowed_labels: Optional[List[str]] = None

    def resolved_model_path(self) -> Optional[str]:
        return self.beats_model_path or self.detector_model_path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "beats_model_path": self.beats_model_path,
            "detector_model_path": self.detector_model_path,
            "score_threshold": self.score_threshold,
            "clap_label_score_threshold": self.clap_label_score_threshold,
            "clap_label_fallback_top1": self.clap_label_fallback_top1,
            "window_size_s": self.window_size_s,
            "hop_size_s": self.hop_size_s,
            "inference_batch_size": self.inference_batch_size,
            "min_event_duration_ms": self.min_event_duration_ms,
            "merge_gap_ms": self.merge_gap_ms,
            "onset_tolerance_ms": self.onset_tolerance_ms,
            "offset_tolerance_ms": self.offset_tolerance_ms,
            "offset_tolerance_ratio": self.offset_tolerance_ratio,
            "allowed_labels": list(self.allowed_labels) if self.allowed_labels is not None else None,
        }


@dataclass
class _NormalizedAnnotations:
    labels: List[List[str]]
    events: Optional[List[List[EventSpan]]]
    has_timestamps: bool


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


def _duration_ms(start_ms: int, end_ms: int) -> int:
    return max(0, int(end_ms) - int(start_ms))


def _normalize_text_label(label: str) -> str:
    return " ".join(str(label).strip().split())


def _map_label(label: str, label_mapping: EventLabelMapper) -> Optional[str]:
    normalized = _normalize_text_label(label)
    if not normalized:
        return None

    if label_mapping is None:
        mapped = normalized
    elif callable(label_mapping):
        mapped = label_mapping(normalized)
    else:
        mapped = label_mapping.get(normalized, normalized)

    if mapped is None:
        return None
    mapped_text = _normalize_text_label(str(mapped))
    return mapped_text or None


def _coerce_event_span(raw_event: Union[EventSpan, Dict[str, Any]], context: str) -> EventSpan:
    if isinstance(raw_event, EventSpan):
        if raw_event.end_ms <= raw_event.start_ms:
            raise ValueError(f"{context} has invalid timestamps.")
        return raw_event

    if not isinstance(raw_event, dict):
        raise ValueError(f"{context} must be an EventSpan or a dict event.")

    label = _normalize_text_label(str(raw_event.get("label", "")))
    start_ms = raw_event.get("start_ms")
    end_ms = raw_event.get("end_ms")
    score = raw_event.get("score")

    if not label:
        raise ValueError(f"{context} is missing label.")
    if start_ms is None or end_ms is None:
        raise ValueError(f"{context} is missing start_ms/end_ms.")

    start_ms = int(round(float(start_ms)))
    end_ms = int(round(float(end_ms)))
    if end_ms <= start_ms:
        raise ValueError(f"{context} has end_ms <= start_ms.")

    return EventSpan(
        label=label,
        start_ms=start_ms,
        end_ms=end_ms,
        score=None if score is None else float(score),
    )


def _coerce_label(raw_item: LabelLike, context: str) -> str:
    if isinstance(raw_item, str):
        label = _normalize_text_label(raw_item)
    elif isinstance(raw_item, EventSpan):
        label = _normalize_text_label(raw_item.label)
    elif isinstance(raw_item, dict):
        label = _normalize_text_label(str(raw_item.get("label", "")))
    else:
        raise ValueError(f"{context} must be a string, EventSpan, or dict with label.")

    if not label:
        raise ValueError(f"{context} is missing label.")
    return label


def _normalize_annotation_batches(
    annotations: Sequence[Sequence[EventLike]],
    *,
    name: str,
    expected_length: int,
) -> _NormalizedAnnotations:
    if len(annotations) != expected_length:
        raise ValueError(f"{name} size mismatch: {len(annotations)} vs {expected_length}")

    label_batches: List[List[str]] = []
    event_batches: List[List[EventSpan]] = []
    saw_timed = False
    saw_untimed = False

    for sample_index, sample_items in enumerate(annotations):
        if sample_items is None:
            label_batches.append([])
            event_batches.append([])
            continue
        if not isinstance(sample_items, (list, tuple)):
            raise ValueError(f"{name}[{sample_index}] must be a list.")

        sample_labels: List[str] = []
        sample_events: List[EventSpan] = []
        for item_index, raw_item in enumerate(sample_items):
            context = f"{name}[{sample_index}][{item_index}]"
            if isinstance(raw_item, EventSpan):
                event = _coerce_event_span(raw_item, context)
                sample_events.append(event)
                sample_labels.append(event.label)
                saw_timed = True
                continue

            if isinstance(raw_item, dict) and raw_item.get("start_ms") is not None and raw_item.get("end_ms") is not None:
                event = _coerce_event_span(raw_item, context)
                sample_events.append(event)
                sample_labels.append(event.label)
                saw_timed = True
                continue

            sample_labels.append(_coerce_label(raw_item, context))
            saw_untimed = True

        sample_events = sorted(sample_events, key=lambda item: (item.start_ms, item.end_ms, item.label))
        label_batches.append(sample_labels)
        event_batches.append(sample_events)

    has_timestamps = saw_timed and not saw_untimed
    return _NormalizedAnnotations(
        labels=label_batches,
        events=event_batches if has_timestamps else None,
        has_timestamps=has_timestamps,
    )


def _filter_reference_labels(labels: Sequence[str], allowed_labels: Optional[Sequence[str]]) -> List[str]:
    allowed = set(allowed_labels) if allowed_labels is not None else None
    filtered: List[str] = []
    for label in labels:
        normalized = _normalize_text_label(label)
        if not normalized:
            continue
        if allowed is not None and normalized not in allowed:
            continue
        filtered.append(normalized)
    return filtered


def _filter_target_labels(
    labels: Sequence[str],
    *,
    label_mapping: EventLabelMapper,
    allowed_labels: Optional[Sequence[str]],
) -> List[str]:
    allowed = set(allowed_labels) if allowed_labels is not None else None
    filtered: List[str] = []
    for label in labels:
        mapped = _map_label(label, label_mapping)
        if mapped is None:
            continue
        if allowed is not None and mapped not in allowed:
            continue
        filtered.append(mapped)
    return filtered


def _filter_reference_events(
    events: Sequence[EventSpan],
    *,
    allowed_labels: Optional[Sequence[str]],
) -> List[EventSpan]:
    allowed = set(allowed_labels) if allowed_labels is not None else None
    filtered: List[EventSpan] = []
    for event in events:
        label = _normalize_text_label(event.label)
        if not label:
            continue
        if allowed is not None and label not in allowed:
            continue
        filtered.append(
            EventSpan(
                label=label,
                start_ms=int(event.start_ms),
                end_ms=int(event.end_ms),
                score=event.score,
            )
        )
    return filtered


def _filter_target_events(
    events: Sequence[EventSpan],
    *,
    label_mapping: EventLabelMapper,
    allowed_labels: Optional[Sequence[str]],
) -> List[EventSpan]:
    allowed = set(allowed_labels) if allowed_labels is not None else None
    filtered: List[EventSpan] = []
    for event in events:
        mapped = _map_label(event.label, label_mapping)
        if mapped is None:
            continue
        if allowed is not None and mapped not in allowed:
            continue
        filtered.append(
            EventSpan(
                label=mapped,
                start_ms=int(event.start_ms),
                end_ms=int(event.end_ms),
                score=event.score,
            )
        )
    return filtered


def _compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _merge_annotation_views(
    primary: _NormalizedAnnotations,
    fallback: _NormalizedAnnotations,
) -> _NormalizedAnnotations:
    if len(primary.labels) != len(fallback.labels):
        raise ValueError("Primary/fallback annotation size mismatch.")

    merged_labels: List[List[str]] = []
    for primary_labels, fallback_labels in zip(primary.labels, fallback.labels):
        merged_labels.append(list(primary_labels) if primary_labels else list(fallback_labels))

    return _NormalizedAnnotations(
        labels=merged_labels,
        events=primary.events,
        has_timestamps=primary.has_timestamps,
    )


def _strict_match(reference_event: EventSpan, predicted_event: EventSpan, config: DiscreteEventConfig) -> bool:
    if reference_event.label != predicted_event.label:
        return False
    if abs(int(reference_event.start_ms) - int(predicted_event.start_ms)) > int(config.onset_tolerance_ms):
        return False

    reference_duration = _duration_ms(reference_event.start_ms, reference_event.end_ms)
    offset_tolerance = max(
        int(config.offset_tolerance_ms),
        int(round(reference_duration * float(config.offset_tolerance_ratio))),
    )
    return abs(int(reference_event.end_ms) - int(predicted_event.end_ms)) <= offset_tolerance


def _maximum_cardinality_matches(
    reference_events: Sequence[EventSpan],
    predicted_events: Sequence[EventSpan],
    config: DiscreteEventConfig,
) -> List[Tuple[int, int]]:
    if not reference_events or not predicted_events:
        return []

    adjacency: List[List[int]] = []
    for ref_event in reference_events:
        candidate_indices = [
            pred_index
            for pred_index, pred_event in enumerate(predicted_events)
            if _strict_match(ref_event, pred_event, config)
        ]
        candidate_indices.sort(
            key=lambda pred_index: (
                abs(predicted_events[pred_index].start_ms - ref_event.start_ms)
                + abs(predicted_events[pred_index].end_ms - ref_event.end_ms),
                predicted_events[pred_index].start_ms,
                predicted_events[pred_index].end_ms,
            )
        )
        adjacency.append(candidate_indices)

    matched_prediction_to_reference = [-1] * len(predicted_events)

    def _dfs(reference_index: int, visited: List[bool]) -> bool:
        for prediction_index in adjacency[reference_index]:
            if visited[prediction_index]:
                continue
            visited[prediction_index] = True
            current_reference = matched_prediction_to_reference[prediction_index]
            if current_reference == -1 or _dfs(current_reference, visited):
                matched_prediction_to_reference[prediction_index] = reference_index
                return True
        return False

    for reference_index in range(len(reference_events)):
        _dfs(reference_index, [False] * len(predicted_events))

    pairs = [
        (reference_index, prediction_index)
        for prediction_index, reference_index in enumerate(matched_prediction_to_reference)
        if reference_index != -1
    ]
    pairs.sort()
    return pairs


def _score_relaxed_f1(
    reference_labels: Sequence[Sequence[str]],
    predicted_labels: Sequence[Sequence[str]],
    *,
    label_mapping: EventLabelMapper,
    allowed_labels: Optional[Sequence[str]],
    sample_ids: Optional[Sequence[str]],
) -> Dict[str, Any]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    sample_details: List[Dict[str, Any]] = []

    for sample_index, (ref_items, pred_items) in enumerate(zip(reference_labels, predicted_labels)):
        ref_labels = _filter_reference_labels(ref_items, allowed_labels)
        pred_labels = _filter_target_labels(
            pred_items,
            label_mapping=label_mapping,
            allowed_labels=allowed_labels,
        )

        ref_counter = Counter(ref_labels)
        pred_counter = Counter(pred_labels)
        all_labels = sorted(set(ref_counter) | set(pred_counter))

        sample_tp = 0
        sample_fp = 0
        sample_fn = 0
        per_label: Dict[str, Dict[str, int]] = {}
        for label in all_labels:
            label_tp = min(ref_counter.get(label, 0), pred_counter.get(label, 0))
            label_fn = max(ref_counter.get(label, 0) - label_tp, 0)
            label_fp = max(pred_counter.get(label, 0) - label_tp, 0)
            sample_tp += label_tp
            sample_fp += label_fp
            sample_fn += label_fn
            per_label[label] = {
                "tp": int(label_tp),
                "fp": int(label_fp),
                "fn": int(label_fn),
            }

        total_tp += sample_tp
        total_fp += sample_fp
        total_fn += sample_fn
        sample_details.append(
            {
                "sample_index": sample_index,
                "sample_id": sample_ids[sample_index] if sample_ids is not None else str(sample_index),
                "reference_labels": ref_labels,
                "predicted_labels": pred_labels,
                "tp": sample_tp,
                "fp": sample_fp,
                "fn": sample_fn,
                "per_label": per_label,
            }
        )

    return {
        "micro": _compute_prf(total_tp, total_fp, total_fn),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "samples": sample_details,
    }


def _score_strict_f1(
    reference_events: Sequence[Sequence[EventSpan]],
    predicted_events: Sequence[Sequence[EventSpan]],
    *,
    config: DiscreteEventConfig,
    label_mapping: EventLabelMapper,
    allowed_labels: Optional[Sequence[str]],
    sample_ids: Optional[Sequence[str]],
) -> Dict[str, Any]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    sample_details: List[Dict[str, Any]] = []

    for sample_index, (ref_items, pred_items) in enumerate(zip(reference_events, predicted_events)):
        ref_events = _filter_reference_events(ref_items, allowed_labels=allowed_labels)
        pred_events = _filter_target_events(
            pred_items,
            label_mapping=label_mapping,
            allowed_labels=allowed_labels,
        )

        ref_by_label: Dict[str, List[Tuple[int, EventSpan]]] = defaultdict(list)
        pred_by_label: Dict[str, List[Tuple[int, EventSpan]]] = defaultdict(list)
        for ref_index, event in enumerate(ref_events):
            ref_by_label[event.label].append((ref_index, event))
        for pred_index, event in enumerate(pred_events):
            pred_by_label[event.label].append((pred_index, event))

        matched_pairs: List[Tuple[int, int]] = []
        for label, ref_label_events in ref_by_label.items():
            pred_label_events = pred_by_label.get(label, [])
            if not pred_label_events:
                continue
            label_pairs = _maximum_cardinality_matches(
                [item[1] for item in ref_label_events],
                [item[1] for item in pred_label_events],
                config,
            )
            for ref_local_index, pred_local_index in label_pairs:
                matched_pairs.append(
                    (
                        ref_label_events[ref_local_index][0],
                        pred_label_events[pred_local_index][0],
                    )
                )

        matched_reference = {pair[0] for pair in matched_pairs}
        matched_prediction = {pair[1] for pair in matched_pairs}
        sample_tp = len(matched_pairs)
        sample_fn = len(ref_events) - sample_tp
        sample_fp = len(pred_events) - sample_tp

        total_tp += sample_tp
        total_fp += sample_fp
        total_fn += sample_fn
        sample_details.append(
            {
                "sample_index": sample_index,
                "sample_id": sample_ids[sample_index] if sample_ids is not None else str(sample_index),
                "reference_events": [event.to_dict() for event in ref_events],
                "predicted_events": [event.to_dict() for event in pred_events],
                "matched_pairs": matched_pairs,
                "tp": sample_tp,
                "fp": sample_fp,
                "fn": sample_fn,
                "unmatched_reference_indices": [
                    index for index in range(len(ref_events)) if index not in matched_reference
                ],
                "unmatched_predicted_indices": [
                    index for index in range(len(pred_events)) if index not in matched_prediction
                ],
            }
        )

    return {
        "micro": _compute_prf(total_tp, total_fp, total_fn),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "samples": sample_details,
    }


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

        raw_events = item.get("events") or []
        if not isinstance(raw_events, list):
            raise ValueError(f"Manifest item {index} has invalid events.")
        events = [
            _coerce_event_span(raw_event, f"manifest[{index}].events[{event_index}]")
            for event_index, raw_event in enumerate(raw_events)
        ]
        events = sorted(events, key=lambda event: (event.start_ms, event.end_ms, event.label))

        raw_labels = item.get("labels")
        if raw_labels is None:
            metadata = item.get("metadata", {})
            raw_labels = metadata.get("labels", []) if isinstance(metadata, dict) else []
        if not isinstance(raw_labels, list):
            raise ValueError(f"Manifest item {index} has invalid labels.")
        labels = [_coerce_label(raw_label, f"manifest[{index}].labels") for raw_label in raw_labels]
        if not labels and events:
            labels = [event.label for event in events]

        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        samples.append(
            ParalinguisticSample(
                sample_id=str(item.get("id", index)),
                source_audio=ensure_existing_audio(str(source_audio)),
                source_text=str(item.get("source_text", "")).strip(),
                events=events,
                labels=labels,
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
        "events": [list(sample.events) for sample in samples],
        "labels": [list(sample.labels) for sample in samples],
        "metadata": [dict(sample.metadata) for sample in samples],
    }


def _load_data_list(input_data: Union[str, List[str]], name: str) -> List[str]:
    if isinstance(input_data, list):
        return [str(item) for item in input_data]
    if not isinstance(input_data, str):
        raise ValueError(f"{name} must be a path or a list of paths.")

    path = Path(input_data)
    if path.is_dir():
        return _load_audio_from_folder(str(path))
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {input_data}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{name} JSON must be a list.")
        return [str(item) for item in data]

    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_audio_from_folder(folder_path: str, extensions: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a")) -> List[str]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    audio_files: List[Path] = []
    for extension in extensions:
        audio_files.extend(folder.glob(f"*{extension}"))
    audio_files = sorted(audio_files, key=lambda item: item.stem)
    if not audio_files:
        raise ValueError(f"No audio files found under: {folder_path}")
    return [str(item.resolve()) for item in audio_files]


class BuiltinAudioEventDetector:
    DEFAULT_MODEL_CANDIDATES = (
        "microsoft/beats-base",
        "microsoft/beats-base-finetuned-audioset",
        "hf-audio/beats-base",
        "hf-audio/beats-base-audioset",
    )

    def __init__(self, config: DiscreteEventConfig, device: Optional[str] = None):
        self.config = config
        self.device = _to_device(device)
        self._feature_extractor = None
        self._model = None
        self._id2label: Optional[Dict[int, str]] = None
        self._loaded_model_name: Optional[str] = None

    def _candidate_model_names(self) -> List[str]:
        configured_path = self.config.resolved_model_path()
        if configured_path:
            return [configured_path]
        return list(self.DEFAULT_MODEL_CANDIDATES)

    def _load_model(self) -> None:
        if self._model is not None and self._feature_extractor is not None and self._id2label is not None:
            return

        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        except ImportError as exc:
            raise RuntimeError("Discrete acoustic event F1 requires `transformers` to load BEATs.") from exc

        load_errors: List[str] = []
        for model_name in self._candidate_model_names():
            try:
                feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
                model = AutoModelForAudioClassification.from_pretrained(model_name).to(self.device).eval()
                raw_id2label = getattr(model.config, "id2label", None)
                if not raw_id2label:
                    raise RuntimeError("checkpoint does not expose id2label")
                self._feature_extractor = feature_extractor
                self._model = model
                self._id2label = {int(index): str(label) for index, label in raw_id2label.items()}
                self._loaded_model_name = model_name
                return
            except Exception as exc:  # noqa: BLE001
                load_errors.append(f"{model_name}: {exc}")

        joined_errors = "\n".join(load_errors)
        raise RuntimeError(
            "Failed to load a BEATs checkpoint. "
            "Pass `beats_model_path`/`detector_model_path` to a local checkpoint or a valid Hugging Face repo.\n"
            f"{joined_errors}"
        )

    def _window_boundaries(self, total_samples: int, sr: int) -> List[Tuple[int, int]]:
        window_size = max(int(round(self.config.window_size_s * sr)), 1)
        hop_size = max(int(round(self.config.hop_size_s * sr)), 1)
        if total_samples <= window_size:
            return [(0, total_samples)]

        boundaries: List[Tuple[int, int]] = []
        start = 0
        while start < total_samples:
            end = min(start + window_size, total_samples)
            boundaries.append((start, end))
            if end >= total_samples:
                break
            start += hop_size
        return boundaries

    def _scores_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        assert self._model is not None
        problem_type = getattr(self._model.config, "problem_type", None)
        if problem_type == "single_label_classification":
            return torch.softmax(logits, dim=-1)
        return torch.sigmoid(logits)

    def _merge_positive_windows(
        self,
        positive_windows: Dict[str, List[Tuple[int, int, float]]],
    ) -> List[EventSpan]:
        merged_events: List[EventSpan] = []
        allowed = set(self.config.allowed_labels) if self.config.allowed_labels is not None else None

        for label, windows in positive_windows.items():
            if allowed is not None and label not in allowed:
                continue
            windows = sorted(windows, key=lambda item: (item[0], item[1]))
            if not windows:
                continue

            current_start, current_end, current_score = windows[0]
            for start_ms, end_ms, score in windows[1:]:
                if start_ms <= current_end + int(self.config.merge_gap_ms):
                    current_end = max(current_end, end_ms)
                    current_score = max(current_score, score)
                else:
                    if _duration_ms(current_start, current_end) >= int(self.config.min_event_duration_ms):
                        merged_events.append(
                            EventSpan(
                                label=label,
                                start_ms=current_start,
                                end_ms=current_end,
                                score=float(current_score),
                            )
                        )
                    current_start, current_end, current_score = start_ms, end_ms, score

            if _duration_ms(current_start, current_end) >= int(self.config.min_event_duration_ms):
                merged_events.append(
                    EventSpan(
                        label=label,
                        start_ms=current_start,
                        end_ms=current_end,
                        score=float(current_score),
                    )
                )

        return sorted(merged_events, key=lambda item: (item.start_ms, item.end_ms, item.label))

    def _detect_single(self, audio_path: str) -> List[EventSpan]:
        self._load_model()
        assert self._feature_extractor is not None
        assert self._model is not None
        assert self._id2label is not None

        sampling_rate = int(getattr(self._feature_extractor, "sampling_rate", 16000) or 16000)
        waveform, sr = _load_audio_mono(audio_path, target_sr=sampling_rate)
        if waveform.size == 0:
            return []

        boundaries = self._window_boundaries(len(waveform), sr)
        batch_size = max(int(self.config.inference_batch_size), 1)
        positive_windows: Dict[str, List[Tuple[int, int, float]]] = defaultdict(list)

        for batch_start in range(0, len(boundaries), batch_size):
            batch_boundaries = boundaries[batch_start : batch_start + batch_size]
            batch_audio = [waveform[start:end] for start, end in batch_boundaries]
            inputs = self._feature_extractor(
                batch_audio,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                logits = self._model(**inputs).logits
            scores = self._scores_from_logits(logits).detach().cpu().numpy()

            for window_index, (start_sample, end_sample) in enumerate(batch_boundaries):
                start_ms = int(round(start_sample * 1000.0 / sr))
                end_ms = int(round(end_sample * 1000.0 / sr))
                for class_index, score in enumerate(scores[window_index]):
                    if float(score) < float(self.config.score_threshold):
                        continue
                    label = self._id2label.get(class_index)
                    if not label:
                        continue
                    positive_windows[str(label)].append((start_ms, end_ms, float(score)))

        return self._merge_positive_windows(positive_windows)

    def detect(self, audio_paths: Sequence[str]) -> List[List[EventSpan]]:
        detections: List[List[EventSpan]] = []
        for audio_path in tqdm(audio_paths, desc="Detecting BEATs events", unit="file"):
            detections.append(self._detect_single(str(audio_path)))
        return detections


class ParalinguisticEvaluator:
    DEFAULT_CLAP_MODEL = "laion/clap-htsat-fused"

    def __init__(
        self,
        use_continuous_fidelity: bool = True,
        use_discrete_event_f1: bool = False,
        clap_model_path: Optional[str] = None,
        beats_model_path: Optional[str] = None,
        discrete_event_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        **_: Any,
    ):
        payload = dict(discrete_event_config or {})
        payload.pop("enabled", None)
        if beats_model_path is not None:
            payload["beats_model_path"] = beats_model_path

        self.device = _to_device(device)
        self.use_continuous_fidelity = bool(use_continuous_fidelity)
        self.use_discrete_event_f1 = bool(use_discrete_event_f1)
        self.clap_model_path = clap_model_path or self.DEFAULT_CLAP_MODEL
        self.discrete_event_config = DiscreteEventConfig(enabled=self.use_discrete_event_f1, **payload)

        self._clap_processor = None
        self._clap_model = None
        self._event_detector: Optional[BuiltinAudioEventDetector] = None

    def _load_clap(self) -> None:
        if self._clap_model is not None and self._clap_processor is not None:
            return

        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise RuntimeError("Paralinguistic_Fidelity_Cosine requires `transformers` to load CLAP.") from exc

        self._clap_processor = ClapProcessor.from_pretrained(self.clap_model_path)
        self._clap_model = ClapModel.from_pretrained(self.clap_model_path).to(self.device).eval()

    def _load_event_detector(self) -> None:
        if self._event_detector is None:
            self._event_detector = BuiltinAudioEventDetector(self.discrete_event_config, device=self.device)

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

    def _extract_clap_text_embeddings(self, texts: Sequence[str]) -> List[np.ndarray]:
        self._load_clap()
        assert self._clap_processor is not None
        assert self._clap_model is not None

        embeddings: List[np.ndarray] = []
        for text in texts:
            inputs = self._clap_processor(text=[str(text)], return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                features = self._clap_model.get_text_features(**inputs)
            embeddings.append(features[0].detach().cpu().numpy())
        return embeddings

    @staticmethod
    def _build_clap_prompts_for_label(label: str) -> List[str]:
        normalized = _normalize_text_label(label)
        return [
            normalized,
            f"the sound of {normalized}",
            f"an audio recording of {normalized}",
            f"a person producing {normalized}",
        ]

    def _predict_labels_with_clap(
        self,
        audio_paths: Sequence[str],
        candidate_labels: Sequence[str],
    ) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
        normalized_candidate_labels = [
            _normalize_text_label(label)
            for label in candidate_labels
            if _normalize_text_label(label)
        ]
        if not normalized_candidate_labels:
            return [[] for _ in audio_paths], [
                {"sample_index": index, "predicted_labels": [], "label_scores": {}}
                for index in range(len(audio_paths))
            ]

        unique_candidate_labels = list(dict.fromkeys(normalized_candidate_labels))
        prompt_records: List[Tuple[str, str]] = []
        for label in unique_candidate_labels:
            for prompt in self._build_clap_prompts_for_label(label):
                prompt_records.append((label, prompt))

        audio_embeddings = self._extract_clap_embeddings(audio_paths)
        text_embeddings = self._extract_clap_text_embeddings([prompt for _, prompt in prompt_records])

        normalized_audio_embeddings = []
        for embedding in audio_embeddings:
            embedding_norm = float(np.linalg.norm(embedding))
            if embedding_norm <= 0.0:
                normalized_audio_embeddings.append(embedding)
            else:
                normalized_audio_embeddings.append(embedding / embedding_norm)

        normalized_text_embeddings = []
        for embedding in text_embeddings:
            embedding_norm = float(np.linalg.norm(embedding))
            if embedding_norm <= 0.0:
                normalized_text_embeddings.append(embedding)
            else:
                normalized_text_embeddings.append(embedding / embedding_norm)

        predicted_labels: List[List[str]] = []
        diagnostics: List[Dict[str, Any]] = []
        threshold = float(self.discrete_event_config.clap_label_score_threshold)
        fallback_top1 = bool(self.discrete_event_config.clap_label_fallback_top1)

        for sample_index, audio_embedding in enumerate(normalized_audio_embeddings):
            label_scores: Dict[str, float] = {}
            for (label, _prompt), text_embedding in zip(prompt_records, normalized_text_embeddings):
                score = float(np.dot(audio_embedding, text_embedding))
                if label not in label_scores or score > label_scores[label]:
                    label_scores[label] = score

            selected = sorted(
                [label for label, score in label_scores.items() if score >= threshold],
                key=lambda item: (-label_scores[item], item),
            )
            if not selected and fallback_top1 and label_scores:
                top_label = max(label_scores.items(), key=lambda item: item[1])[0]
                selected = [top_label]

            predicted_labels.append(selected)
            diagnostics.append(
                {
                    "sample_index": sample_index,
                    "predicted_labels": selected,
                    "label_scores": {label: round(score, 4) for label, score in sorted(label_scores.items())},
                }
            )

        return predicted_labels, diagnostics

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

    def _resolve_reference_annotations(
        self,
        *,
        source_event_annotations: Optional[Sequence[Sequence[EventLike]]],
        source_utterance_annotations: Optional[Sequence[Sequence[LabelLike]]],
        num_samples: int,
    ) -> _NormalizedAnnotations:
        if source_event_annotations is not None:
            event_annotations = _normalize_annotation_batches(
                source_event_annotations,
                name="source_event_annotations",
                expected_length=num_samples,
            )
            if source_utterance_annotations is None:
                return event_annotations
            label_annotations = _normalize_annotation_batches(
                source_utterance_annotations,
                name="source_utterance_annotations",
                expected_length=num_samples,
            )
            return _merge_annotation_views(event_annotations, label_annotations)
        if source_utterance_annotations is not None:
            return _normalize_annotation_batches(
                source_utterance_annotations,
                name="source_utterance_annotations",
                expected_length=num_samples,
            )
        raise ValueError(
            "Discrete acoustic event F1 requires `source_event_annotations` or "
            "`source_utterance_annotations`."
        )

    def _resolve_predicted_annotations(
        self,
        *,
        target_audio_paths: Sequence[str],
        target_event_annotations: Optional[Sequence[Sequence[EventLike]]],
        target_utterance_annotations: Optional[Sequence[Sequence[LabelLike]]],
        num_samples: int,
    ) -> Tuple[_NormalizedAnnotations, str, Optional[List[Dict[str, Any]]]]:
        if target_event_annotations is not None:
            normalized = _normalize_annotation_batches(
                target_event_annotations,
                name="target_event_annotations",
                expected_length=num_samples,
            )
            if target_utterance_annotations is not None:
                label_annotations = _normalize_annotation_batches(
                    target_utterance_annotations,
                    name="target_utterance_annotations",
                    expected_length=num_samples,
                )
                normalized = _merge_annotation_views(normalized, label_annotations)
            return normalized, "provided_target_event_annotations", None

        if target_utterance_annotations is not None:
            normalized = _normalize_annotation_batches(
                target_utterance_annotations,
                name="target_utterance_annotations",
                expected_length=num_samples,
            )
            return normalized, "provided_target_utterance_annotations", None

        self._load_event_detector()
        assert self._event_detector is not None
        detections = self._event_detector.detect(target_audio_paths)
        normalized = _normalize_annotation_batches(
            detections,
            name="beats_detections",
            expected_length=num_samples,
        )
        return normalized, "beats_detector", None

    def evaluate_all(
        self,
        source_audio: Union[List[str], str],
        target_audio: Union[List[str], str],
        *,
        source_event_annotations: Optional[Sequence[Sequence[EventLike]]] = None,
        target_event_annotations: Optional[Sequence[Sequence[EventLike]]] = None,
        source_utterance_annotations: Optional[Sequence[Sequence[LabelLike]]] = None,
        target_utterance_annotations: Optional[Sequence[Sequence[LabelLike]]] = None,
        event_label_mapping: EventLabelMapper = None,
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

        if self.use_discrete_event_f1:
            reference_annotations = self._resolve_reference_annotations(
                source_event_annotations=source_event_annotations,
                source_utterance_annotations=source_utterance_annotations,
                num_samples=num_samples,
            )
            reference_label_vocab = sorted(
                {
                    label
                    for sample_labels in reference_annotations.labels
                    for label in _filter_reference_labels(sample_labels, None)
                }
            )
            allowed_labels = (
                list(self.discrete_event_config.allowed_labels)
                if self.discrete_event_config.allowed_labels is not None
                else reference_label_vocab
            )
            if not allowed_labels:
                allowed_labels = None

            clap_label_diagnostics: Optional[List[Dict[str, Any]]] = None
            if (
                not reference_annotations.has_timestamps
                and target_event_annotations is None
                and target_utterance_annotations is None
            ):
                clap_candidate_labels = list(allowed_labels or [])
                clap_predictions, clap_label_diagnostics = self._predict_labels_with_clap(
                    target_audio_paths,
                    clap_candidate_labels,
                )
                predicted_annotations = _normalize_annotation_batches(
                    clap_predictions,
                    name="clap_label_predictions",
                    expected_length=num_samples,
                )
                prediction_source = "clap_label_matching"
            else:
                predicted_annotations, prediction_source, clap_label_diagnostics = self._resolve_predicted_annotations(
                    target_audio_paths=target_audio_paths,
                    target_event_annotations=target_event_annotations,
                    target_utterance_annotations=target_utterance_annotations,
                    num_samples=num_samples,
                )

            relaxed = _score_relaxed_f1(
                reference_annotations.labels,
                predicted_annotations.labels,
                label_mapping=event_label_mapping,
                allowed_labels=allowed_labels,
                sample_ids=sample_ids,
            )
            results["Discrete_Acoustic_Event_F1_Relaxed"] = relaxed["micro"]["f1"]

            strict = None
            if reference_annotations.has_timestamps and predicted_annotations.has_timestamps:
                assert reference_annotations.events is not None
                assert predicted_annotations.events is not None
                strict = _score_strict_f1(
                    reference_annotations.events,
                    predicted_annotations.events,
                    config=self.discrete_event_config,
                    label_mapping=event_label_mapping,
                    allowed_labels=allowed_labels,
                    sample_ids=sample_ids,
                )
                results["Discrete_Acoustic_Event_F1_Strict"] = strict["micro"]["f1"]

            diagnostics["discrete_event_metrics"] = {
                "config": self.discrete_event_config.to_dict(),
                "prediction_source": prediction_source,
                "reference_has_timestamps": reference_annotations.has_timestamps,
                "predicted_has_timestamps": predicted_annotations.has_timestamps,
                "clap_label_matching": {
                    "score_threshold": self.discrete_event_config.clap_label_score_threshold,
                    "fallback_top1": self.discrete_event_config.clap_label_fallback_top1,
                    "candidate_labels": list(allowed_labels or []),
                    "samples": clap_label_diagnostics,
                }
                if prediction_source == "clap_label_matching"
                else None,
                "relaxed": relaxed,
                "strict": strict,
                "output_metrics": list(results.keys()),
            }

        if verbose:
            print("\n[ParalinguisticEvaluator] Summary")
            print(f"  Samples: {num_samples}")
            for metric_name, score in results.items():
                print(f"  {metric_name}: {score}")

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
    target_event_annotations: Optional[Sequence[Sequence[EventLike]]] = None,
    target_utterance_annotations: Optional[Sequence[Sequence[LabelLike]]] = None,
    event_label_mapping: EventLabelMapper = None,
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
        evaluator = ParalinguisticEvaluator(device=device, **final_kwargs)

    has_source_events = any(len(sample_events) > 0 for sample_events in inputs["events"])
    has_source_labels = any(len(sample_labels) > 0 for sample_labels in inputs["labels"])

    result = evaluator.evaluate_all(
        source_audio=inputs["source_audio"],
        target_audio=target_audio,
        source_event_annotations=inputs["events"] if has_source_events else None,
        source_utterance_annotations=inputs["labels"] if has_source_labels else None,
        target_event_annotations=target_event_annotations,
        target_utterance_annotations=target_utterance_annotations,
        event_label_mapping=event_label_mapping,
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
