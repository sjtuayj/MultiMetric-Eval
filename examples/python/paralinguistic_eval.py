from multimetric_eval import ParalinguisticEvaluator


LABEL_MAP = {
    "laugh": "laughter",
    "laughing": "laughter",
    "laughter": "laughter",
    "sigh": "sighing",
    "sighing": "sighing",
    "clear throat": "throat clearing",
    "clearing throat": "throat clearing",
    "throat clearing": "throat clearing",
}

CANDIDATE_LABELS = [
    "laughter",
    "sighing",
    "throat clearing",
]


def normalize_label(label: str):
    normalized = " ".join(str(label).strip().lower().replace("_", " ").replace("-", " ").split())
    return LABEL_MAP.get(normalized, normalized)


def main():
    evaluator = ParalinguisticEvaluator(
        use_continuous_fidelity=True,
        use_event_preservation=True,
        clap_model_path="./model/clap-htsat-fused",  # Or "laion/clap-htsat-fused"
        event_prediction_config={
            "score_threshold": 0.2,
            "fallback_top1": False,
        },
        device="cuda",
    )

    scores, diagnostics = evaluator.evaluate_all(
        source_audio=["./src_wavs/sample_001.wav", "./src_wavs/sample_002.wav"],
        target_audio=["./tgt_wavs/sample_001.wav", "./tgt_wavs/sample_002.wav"],
        source_labels=["laugh", "throat clearing"],
        candidate_labels=CANDIDATE_LABELS,
        label_normalizer=normalize_label,
        sample_ids=["sample_001", "sample_002"],
        return_diagnostics=True,
    )

    print(scores)
    print(diagnostics["event_preservation"]["per_label"])


if __name__ == "__main__":
    main()
