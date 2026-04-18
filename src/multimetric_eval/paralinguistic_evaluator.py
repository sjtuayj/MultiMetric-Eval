import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from numpy.linalg import norm
from tqdm import tqdm

try:
    import librosa
except ImportError:
    librosa = None


EventLike = Union["EventSpan", Dict[str, Any]]
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
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }
        if self.score is not None:
            payload["score"] = self.score
        return payload


@dataclass(frozen=True)
class ParalinguisticSample:
    sample_id: str
    source_audio: str
    source_text: str = ""
    events: List[EventSpan] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_event_annotation(self) -> List[Dict[str, Any]]:
        return [event.to_dict() for event in self.events]


@dataclass(frozen=True)
class DiscreteEventConfig:
    enabled: bool = False
    detector_backend: str = "panns"
    detector_model_path: Optional[str] = None
    score_threshold: float = 0.3
    min_event_duration_ms: int = 100
    merge_gap_ms: int = 150
    onset_tolerance_ms: int = 200
    offset_tolerance_ms: int = 200
    offset_tolerance_ratio: float = 0.2
    allowed_labels: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "detector_backend": self.detector_backend,
            "detector_model_path": self.detector_model_path,
            "score_threshold": self.score_threshold,
            "min_event_duration_ms": self.min_event_duration_ms,
            "merge_gap_ms": self.merge_gap_ms,
            "onset_tolerance_ms": self.onset_tolerance_ms,
            "offset_tolerance_ms": self.offset_tolerance_ms,
            "offset_tolerance_ratio": self.offset_tolerance_ratio,
            "allowed_labels": list(self.allowed_labels) if self.allowed_labels is not None else None,
        }


def ensure_existing_audio(path: str) -> str:
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {path}")
    return str(audio_path.resolve())


def _require_librosa() -> None:
    if librosa is None:
        raise RuntimeError(
            "Paralinguistic audio processing requires `librosa`. "
            "Install `multimetriceval[paralinguistics]` or `librosa`."
        )


def _coerce_event_span(raw_event: EventLike, context: str) -> EventSpan:
    if isinstance(raw_event, EventSpan):
        if raw_event.end_ms <= raw_event.start_ms:
            raise ValueError(f"{context} has an invalid event span.")
        return raw_event

    if not isinstance(raw_event, dict):
        raise ValueError(f"{context} must contain EventSpan objects or dict events.")

    label = raw_event.get("label")
    start_ms = raw_event.get("start_ms")
    end_ms = raw_event.get("end_ms")
    score = raw_event.get("score")

    if label is None or start_ms is None or end_ms is None:
        raise ValueError(f"{context} is missing label/start_ms/end_ms.")

    start_ms = int(start_ms)
    end_ms = int(end_ms)
    if end_ms <= start_ms:
        raise ValueError(f"{context} has end_ms <= start_ms.")

    if score is not None:
        score = float(score)

    return EventSpan(label=str(label), start_ms=start_ms, end_ms=end_ms, score=score)


def _coerce_event_batches(
    event_batches: Sequence[Sequence[EventLike]],
    name: str,
    expected_length: Optional[int] = None,
) -> List[List[EventSpan]]:
    if expected_length is not None and len(event_batches) != expected_length:
        raise ValueError(f"{name} size mismatch: {len(event_batches)} vs {expected_length}")

    normalized_batches: List[List[EventSpan]] = []
    for sample_index, events in enumerate(event_batches):
        if events is None:
            normalized_batches.append([])
            continue
        if not isinstance(events, (list, tuple)):
            raise ValueError(f"{name}[{sample_index}] must be a list of events.")

        normalized_sample: List[EventSpan] = []
        for event_index, raw_event in enumerate(events):
            normalized_sample.append(
                _coerce_event_span(raw_event, f"{name}[{sample_index}][{event_index}]")
            )
        normalized_batches.append(sorted(normalized_sample, key=lambda event: (event.start_ms, event.end_ms, event.label)))

    return normalized_batches


def _map_target_label(label: str, label_mapping: EventLabelMapper) -> Optional[str]:
    mapped: Optional[str]
    if label_mapping is None:
        mapped = label
    elif callable(label_mapping):
        mapped = label_mapping(label)
    else:
        mapped = label_mapping.get(label, label)

    if mapped is None:
        return None

    mapped_text = str(mapped).strip()
    if not mapped_text:
        return None
    return mapped_text


def _normalize_target_events(
    events: Sequence[EventSpan],
    *,
    label_mapping: EventLabelMapper,
    allowed_labels: Optional[Sequence[str]],
) -> List[EventSpan]:
    allowed = set(allowed_labels) if allowed_labels is not None else None
    normalized: List[EventSpan] = []

    for event in events:
        mapped_label = _map_target_label(event.label, label_mapping)
        if mapped_label is None:
            continue
        if allowed is not None and mapped_label not in allowed:
            continue
        normalized.append(
            EventSpan(
                label=mapped_label,
                start_ms=int(event.start_ms),
                end_ms=int(event.end_ms),
                score=event.score,
            )
        )

    return sorted(normalized, key=lambda item: (item.start_ms, item.end_ms, item.label))


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
        for event_index, raw_event in enumerate(raw_events):
            try:
                events.append(_coerce_event_span(raw_event, f"manifest[{index}].events[{event_index}]"))
            except ValueError:
                continue

        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        samples.append(
            ParalinguisticSample(
                sample_id=str(item.get("id", index)),
                source_audio=ensure_existing_audio(str(source_audio)),
                source_text=str(item.get("source_text", "")).strip(),
                events=sorted(events, key=lambda event: (event.start_ms, event.end_ms, event.label)),
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

    audio_files = sorted(audio_files, key=lambda item: item.stem)
    if not audio_files:
        raise ValueError(f"No audio files found under: {folder_path}")

    return [str(item) for item in audio_files]


def _event_duration_ms(event: EventSpan) -> int:
    return max(0, int(event.end_ms) - int(event.start_ms))


def _merge_adjacent_events(
    events: Sequence[EventSpan],
    *,
    min_event_duration_ms: int,
    merge_gap_ms: int,
) -> List[EventSpan]:
    if not events:
        return []

    merged: List[EventSpan] = []
    for event in sorted(events, key=lambda item: (item.label, item.start_ms, item.end_ms)):
        if _event_duration_ms(event) < min_event_duration_ms:
            continue

        if (
            merged
            and merged[-1].label == event.label
            and event.start_ms <= merged[-1].end_ms + merge_gap_ms
        ):
            last_event = merged[-1]
            merged[-1] = EventSpan(
                label=last_event.label,
                start_ms=last_event.start_ms,
                end_ms=max(last_event.end_ms, event.end_ms),
                score=max(
                    last_event.score if last_event.score is not None else float("-inf"),
                    event.score if event.score is not None else float("-inf"),
                )
                if last_event.score is not None or event.score is not None
                else None,
            )
        else:
            merged.append(event)

    return merged


def _compute_precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _strict_events_match(reference_event: EventSpan, predicted_event: EventSpan, config: DiscreteEventConfig) -> bool:
    if reference_event.label != predicted_event.label:
        return False

    if abs(reference_event.start_ms - predicted_event.start_ms) > config.onset_tolerance_ms:
        return False

    duration_ms = _event_duration_ms(reference_event)
    offset_tolerance = max(
        config.offset_tolerance_ms,
        int(round(duration_ms * config.offset_tolerance_ratio)),
    )
    return abs(reference_event.end_ms - predicted_event.end_ms) <= offset_tolerance


def _match_events_max_cardinality(
    reference_events: Sequence[EventSpan],
    predicted_events: Sequence[EventSpan],
    pair_ok: Callable[[EventSpan, EventSpan], bool],
) -> List[Tuple[int, int]]:
    if not reference_events or not predicted_events:
        return []

    adjacency: List[List[int]] = []
    for ref_event in reference_events:
        feasible = [
            pred_index
            for pred_index, pred_event in enumerate(predicted_events)
            if pair_ok(ref_event, pred_event)
        ]
        feasible.sort(
            key=lambda pred_index: (
                abs(predicted_events[pred_index].start_ms - ref_event.start_ms)
                + abs(predicted_events[pred_index].end_ms - ref_event.end_ms),
                predicted_events[pred_index].start_ms,
            )
        )
        adjacency.append(feasible)

    matched_prediction_to_reference = [-1] * len(predicted_events)

    def _search(reference_index: int, visited_predictions: List[bool]) -> bool:
        for prediction_index in adjacency[reference_index]:
            if visited_predictions[prediction_index]:
                continue
            visited_predictions[prediction_index] = True
            current_match = matched_prediction_to_reference[prediction_index]
            if current_match == -1 or _search(current_match, visited_predictions):
                matched_prediction_to_reference[prediction_index] = reference_index
                return True
        return False

    for reference_index in range(len(reference_events)):
        _search(reference_index, [False] * len(predicted_events))

    matched_pairs = [
        (reference_index, prediction_index)
        for prediction_index, reference_index in enumerate(matched_prediction_to_reference)
        if reference_index != -1
    ]
    matched_pairs.sort()
    return matched_pairs


def _score_strict_events(
    reference_events: Sequence[EventSpan],
    predicted_events: Sequence[EventSpan],
    config: DiscreteEventConfig,
) -> Dict[str, Any]:
    matched_pairs: List[Tuple[int, int]] = []
    reference_by_label: Dict[str, List[Tuple[int, EventSpan]]] = defaultdict(list)
    predicted_by_label: Dict[str, List[Tuple[int, EventSpan]]] = defaultdict(list)

    for reference_index, event in enumerate(reference_events):
        reference_by_label[event.label].append((reference_index, event))
    for predicted_index, event in enumerate(predicted_events):
        predicted_by_label[event.label].append((predicted_index, event))

    for label, ref_items in reference_by_label.items():
        pred_items = predicted_by_label.get(label, [])
        if not pred_items:
            continue

        label_level_matches = _match_events_max_cardinality(
            [event for _, event in ref_items],
            [event for _, event in pred_items],
            lambda ref_event, pred_event: _strict_events_match(ref_event, pred_event, config),
        )
        for ref_local_index, pred_local_index in label_level_matches:
            matched_pairs.append((ref_items[ref_local_index][0], pred_items[pred_local_index][0]))

    matched_reference = {ref_index for ref_index, _ in matched_pairs}
    matched_predicted = {pred_index for _, pred_index in matched_pairs}
    tp = len(matched_pairs)
    fn = len(reference_events) - tp
    fp = len(predicted_events) - tp

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_pairs": matched_pairs,
        "unmatched_reference_indices": [index for index in range(len(reference_events)) if index not in matched_reference],
        "unmatched_predicted_indices": [index for index in range(len(predicted_events)) if index not in matched_predicted],
    }


def _score_relaxed_events(
    reference_events: Sequence[EventSpan],
    predicted_events: Sequence[EventSpan],
) -> Dict[str, Any]:
    reference_counts: Dict[str, int] = defaultdict(int)
    predicted_counts: Dict[str, int] = defaultdict(int)

    for event in reference_events:
        reference_counts[event.label] += 1
    for event in predicted_events:
        predicted_counts[event.label] += 1

    labels = sorted(set(reference_counts) | set(predicted_counts))
    per_label: Dict[str, Dict[str, int]] = {}
    tp = 0
    fp = 0
    fn = 0

    for label in labels:
        label_tp = min(reference_counts.get(label, 0), predicted_counts.get(label, 0))
        label_fn = max(reference_counts.get(label, 0) - label_tp, 0)
        label_fp = max(predicted_counts.get(label, 0) - label_tp, 0)
        tp += label_tp
        fn += label_fn
        fp += label_fp
        per_label[label] = {"tp": label_tp, "fp": label_fp, "fn": label_fn}

    return {"tp": tp, "fp": fp, "fn": fn, "per_label": per_label}


def _aggregate_event_metrics(
    sample_level_stats: Sequence[Dict[str, Any]],
    label_set: Sequence[str],
) -> Dict[str, Any]:
    total_tp = sum(item["tp"] for item in sample_level_stats)
    total_fp = sum(item["fp"] for item in sample_level_stats)
    total_fn = sum(item["fn"] for item in sample_level_stats)

    per_label_counts: Dict[str, Dict[str, int]] = {label: {"tp": 0, "fp": 0, "fn": 0} for label in label_set}
    for sample_stats in sample_level_stats:
        sample_per_label = sample_stats.get("per_label", {})
        for label, counts in sample_per_label.items():
            if label not in per_label_counts:
                per_label_counts[label] = {"tp": 0, "fp": 0, "fn": 0}
            per_label_counts[label]["tp"] += int(counts.get("tp", 0))
            per_label_counts[label]["fp"] += int(counts.get("fp", 0))
            per_label_counts[label]["fn"] += int(counts.get("fn", 0))

    per_label_metrics: Dict[str, Dict[str, Any]] = {}
    macro_f1_values: List[float] = []
    for label in sorted(per_label_counts):
        counts = per_label_counts[label]
        metrics = _compute_precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])
        support = counts["tp"] + counts["fn"]
        per_label_metrics[label] = {
            **counts,
            **metrics,
            "support": support,
        }
        macro_f1_values.append(metrics["f1"])

    macro_f1 = round(float(sum(macro_f1_values) / len(macro_f1_values)), 4) if macro_f1_values else 0.0

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "micro": _compute_precision_recall_f1(total_tp, total_fp, total_fn),
        "macro_f1": macro_f1,
        "per_label": per_label_metrics,
    }


def _score_discrete_event_batches(
    reference_batches: Sequence[Sequence[EventSpan]],
    predicted_batches: Sequence[Sequence[EventSpan]],
    *,
    config: DiscreteEventConfig,
    label_mapping: EventLabelMapper,
    allowed_labels: Optional[Sequence[str]],
    sample_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    strict_sample_stats: List[Dict[str, Any]] = []
    relaxed_sample_stats: List[Dict[str, Any]] = []
    sample_details: List[Dict[str, Any]] = []
    evaluated_samples = 0
    skipped_samples = 0

    for sample_index, (reference_events, predicted_events) in enumerate(zip(reference_batches, predicted_batches)):
        normalized_predictions = _normalize_target_events(
            predicted_events,
            label_mapping=label_mapping,
            allowed_labels=allowed_labels,
        )
        reference_sorted = sorted(reference_events, key=lambda event: (event.start_ms, event.end_ms, event.label))

        if not reference_sorted and not normalized_predictions:
            skipped_samples += 1
            sample_details.append(
                {
                    "sample_index": sample_index,
                    "sample_id": sample_ids[sample_index] if sample_ids is not None else str(sample_index),
                    "skipped": True,
                    "reason": "no_reference_or_predicted_events",
                }
            )
            continue

        strict_stats = _score_strict_events(reference_sorted, normalized_predictions, config)
        relaxed_stats = _score_relaxed_events(reference_sorted, normalized_predictions)

        per_label_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for reference_event in reference_sorted:
            per_label_counts[reference_event.label]["fn"] += 1
        for ref_index, pred_index in strict_stats["matched_pairs"]:
            label = reference_sorted[ref_index].label
            per_label_counts[label]["tp"] += 1
            per_label_counts[label]["fn"] -= 1
        for pred_index, predicted_event in enumerate(normalized_predictions):
            if pred_index in strict_stats["unmatched_predicted_indices"]:
                per_label_counts[predicted_event.label]["fp"] += 1

        strict_stats["per_label"] = dict(per_label_counts)

        strict_sample_stats.append(strict_stats)
        relaxed_sample_stats.append(relaxed_stats)
        evaluated_samples += 1

        sample_details.append(
            {
                "sample_index": sample_index,
                "sample_id": sample_ids[sample_index] if sample_ids is not None else str(sample_index),
                "skipped": False,
                "reference_events": [event.to_dict() for event in reference_sorted],
                "predicted_events": [event.to_dict() for event in normalized_predictions],
                "strict": {
                    "tp": strict_stats["tp"],
                    "fp": strict_stats["fp"],
                    "fn": strict_stats["fn"],
                    "matched_pairs": strict_stats["matched_pairs"],
                },
                "relaxed": {
                    "tp": relaxed_stats["tp"],
                    "fp": relaxed_stats["fp"],
                    "fn": relaxed_stats["fn"],
                },
            }
        )

    strict_aggregate = _aggregate_event_metrics(strict_sample_stats, allowed_labels or [])
    relaxed_aggregate = _aggregate_event_metrics(relaxed_sample_stats, allowed_labels or [])

    return {
        "evaluated_samples": evaluated_samples,
        "skipped_samples": skipped_samples,
        "strict": strict_aggregate,
        "relaxed": relaxed_aggregate,
        "samples": sample_details,
    }


class BuiltinAudioEventDetector:
    DEFAULT_BACKEND = "panns"
    DEFAULT_TARGET_SR = 32000

    def __init__(self, config: DiscreteEventConfig, device: Optional[str] = None):
        self.config = config
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self._detector = None
        self._panns_labels: Optional[List[str]] = None

    def _load_panns_detector(self) -> None:
        if self._detector is not None:
            return

        try:
            from panns_inference import SoundEventDetection, labels
        except ImportError as exc:
            raise RuntimeError(
                "Discrete paralinguistic event detection requires `panns-inference`. "
                "Install `multimetriceval[paralinguistics]` or `panns-inference`."
            ) from exc

        self._detector = SoundEventDetection(checkpoint_path=self.config.detector_model_path, device=self.device)
        self._panns_labels = list(labels)

    def _detect_with_panns(self, audio_path: str) -> List[EventSpan]:
        self._load_panns_detector()
        assert self._detector is not None
        assert self._panns_labels is not None

        _require_librosa()
        audio_array, _ = librosa.load(audio_path, sr=self.DEFAULT_TARGET_SR, mono=True)
        framewise_output = self._detector.inference(audio_array[None, :])

        if isinstance(framewise_output, tuple):
            candidate_arrays = [np.asarray(item) for item in framewise_output if hasattr(item, "shape")]
            framewise_output = None
            for candidate in candidate_arrays:
                if candidate.ndim == 3 and candidate.shape[-1] == len(self._panns_labels):
                    framewise_output = candidate
                    break
            if framewise_output is None:
                for candidate in candidate_arrays:
                    if candidate.ndim == 2 and candidate.shape[-1] == len(self._panns_labels):
                        framewise_output = candidate
                        break
            if framewise_output is None:
                raise RuntimeError(f"Could not locate framewise PANNs output for {audio_path}.")
        framewise_output = np.asarray(framewise_output)
        if framewise_output.ndim == 3:
            framewise_output = framewise_output[0]
        if framewise_output.ndim != 2:
            raise RuntimeError(
                f"Unexpected PANNs framewise output shape for {audio_path}: {tuple(framewise_output.shape)}"
            )

        num_frames, num_classes = framewise_output.shape
        if num_classes != len(self._panns_labels):
            raise RuntimeError(
                f"PANNs label size mismatch: framewise classes={num_classes}, labels={len(self._panns_labels)}"
            )

        if num_frames == 0:
            return []

        audio_duration_ms = int(round((len(audio_array) / self.DEFAULT_TARGET_SR) * 1000))
        frame_resolution_ms = max(1.0, audio_duration_ms / float(num_frames))

        raw_events: List[EventSpan] = []
        for class_index, label in enumerate(self._panns_labels):
            class_scores = framewise_output[:, class_index]
            active_frames = class_scores >= self.config.score_threshold
            start_index: Optional[int] = None

            for frame_index, is_active in enumerate(active_frames):
                if is_active and start_index is None:
                    start_index = frame_index
                elif not is_active and start_index is not None:
                    end_index = frame_index
                    segment_scores = class_scores[start_index:end_index]
                    raw_events.append(
                        EventSpan(
                            label=label,
                            start_ms=int(round(start_index * frame_resolution_ms)),
                            end_ms=int(round(end_index * frame_resolution_ms)),
                            score=float(np.max(segment_scores)) if len(segment_scores) > 0 else None,
                        )
                    )
                    start_index = None

            if start_index is not None:
                segment_scores = class_scores[start_index:]
                raw_events.append(
                    EventSpan(
                        label=label,
                        start_ms=int(round(start_index * frame_resolution_ms)),
                        end_ms=audio_duration_ms,
                        score=float(np.max(segment_scores)) if len(segment_scores) > 0 else None,
                    )
                )

        return _merge_adjacent_events(
            raw_events,
            min_event_duration_ms=self.config.min_event_duration_ms,
            merge_gap_ms=self.config.merge_gap_ms,
        )

    def detect(self, audio_paths: List[str]) -> List[List[EventSpan]]:
        backend = (self.config.detector_backend or self.DEFAULT_BACKEND).lower()
        if backend != "panns":
            raise ValueError(
                f"Unsupported discrete event detector backend: {self.config.detector_backend}. "
                "Currently only `panns` is implemented."
            )

        detections: List[List[EventSpan]] = []
        for audio_path in tqdm(audio_paths, desc="Detecting acoustic events", unit="file"):
            detections.append(self._detect_with_panns(audio_path))
        return detections


class ParalinguisticEvaluator:
    DEFAULT_CLAP_MODEL = "laion/clap-htsat-fused"

    def __init__(
        self,
        use_continuous_fidelity: bool = True,
        use_discrete_event_f1: bool = False,
        clap_model_path: Optional[str] = None,
        discrete_event_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        discrete_config_payload = dict(discrete_event_config or {})
        discrete_config_payload.pop("enabled", None)

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_continuous_fidelity = use_continuous_fidelity
        self.use_discrete_event_f1 = use_discrete_event_f1
        self.clap_model_path = clap_model_path or self.DEFAULT_CLAP_MODEL
        self.discrete_event_config = DiscreteEventConfig(
            enabled=use_discrete_event_f1,
            **discrete_config_payload,
        )

        self.clap_processor = None
        self.clap_model = None
        self.event_detector: Optional[BuiltinAudioEventDetector] = None

    def _load_clap_model(self) -> None:
        if self.clap_model is not None:
            return

        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise RuntimeError("ParalinguisticEvaluator requires transformers to load CLAP.") from exc

        self.clap_processor = ClapProcessor.from_pretrained(self.clap_model_path)
        self.clap_model = ClapModel.from_pretrained(self.clap_model_path).to(self.device).eval()

    def _load_event_detector(self) -> None:
        if self.event_detector is None:
            self.event_detector = BuiltinAudioEventDetector(self.discrete_event_config, device=self.device)

    def _extract_clap_embeddings(self, audio_paths: List[str]) -> List[np.ndarray]:
        self._load_clap_model()
        assert self.clap_processor is not None
        assert self.clap_model is not None
        _require_librosa()

        results: List[np.ndarray] = []
        for path in tqdm(audio_paths, desc="Extracting CLAP embeddings", unit="file"):
            audio_array, sr = librosa.load(path, sr=48000)
            inputs = self.clap_processor(audio=audio_array, return_tensors="pt", sampling_rate=sr)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                audio_features = self.clap_model.get_audio_features(**inputs)
            results.append(audio_features[0].detach().cpu().numpy())

        return results

    def evaluate_all(
        self,
        source_audio: Union[List[str], str],
        target_audio: Union[List[str], str],
        *,
        source_event_annotations: Optional[Sequence[Sequence[EventLike]]] = None,
        target_event_annotations: Optional[Sequence[Sequence[EventLike]]] = None,
        event_label_mapping: EventLabelMapper = None,
        sample_ids: Optional[Sequence[str]] = None,
        verbose: bool = True,
        return_diagnostics: bool = False,
    ) -> Union[Tuple[Dict[str, float], Dict[str, Any]], Dict[str, float]]:
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

        if self.use_discrete_event_f1:
            if source_event_annotations is None:
                raise ValueError(
                    "source_event_annotations are required when use_discrete_event_f1=True. "
                    "Provide per-sample event spans from the source annotations."
                )

            reference_events = _coerce_event_batches(
                source_event_annotations,
                name="source_event_annotations",
                expected_length=num_samples,
            )
            reference_label_set = sorted(
                set(self.discrete_event_config.allowed_labels or [])
                | {event.label for sample_events in reference_events for event in sample_events}
            )
            allowed_labels_for_scoring: Optional[List[str]] = reference_label_set if reference_label_set else None

            if target_event_annotations is not None:
                predicted_events = _coerce_event_batches(
                    target_event_annotations,
                    name="target_event_annotations",
                    expected_length=num_samples,
                )
            else:
                self._load_event_detector()
                assert self.event_detector is not None
                predicted_events = self.event_detector.detect(tgt_paths)

            discrete_metrics = _score_discrete_event_batches(
                reference_events,
                predicted_events,
                config=self.discrete_event_config,
                label_mapping=event_label_mapping,
                allowed_labels=allowed_labels_for_scoring,
                sample_ids=sample_ids,
            )
            results["Discrete_Acoustic_Event_F1_Strict"] = discrete_metrics["strict"]["micro"]["f1"]
            results["Discrete_Acoustic_Event_F1_Relaxed"] = discrete_metrics["relaxed"]["micro"]["f1"]
            diagnostics["discrete_event_metrics"] = {
                "config": self.discrete_event_config.to_dict(),
                **discrete_metrics,
            }

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
    target_event_annotations: Optional[Sequence[Sequence[EventLike]]] = None,
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

    inputs = build_paralinguistic_inputs(samples)
    if len(inputs["source_audio"]) != len(target_audio):
        raise ValueError(
            f"Source/target size mismatch for paralinguistic evaluation: "
            f"{len(inputs['source_audio'])} vs {len(target_audio)}"
        )

    if evaluator is None:
        final_kwargs = dict(evaluator_kwargs or {})
        final_kwargs.setdefault("use_continuous_fidelity", True)
        evaluator = ParalinguisticEvaluator(
            device=device,
            **final_kwargs,
        )

    result = evaluator.evaluate_all(
        source_audio=inputs["source_audio"],
        target_audio=target_audio,
        source_event_annotations=inputs["events"],
        target_event_annotations=target_event_annotations,
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
