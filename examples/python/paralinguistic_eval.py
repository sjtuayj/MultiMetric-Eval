from multimetric_eval import ParalinguisticEvaluator


def main():
    evaluator = ParalinguisticEvaluator(
        use_continuous_fidelity=True,
        use_discrete_event_f1=True,
        discrete_event_config={
            "detector_backend": "panns",
            "score_threshold": 0.3,
            "onset_tolerance_ms": 200,
            "offset_tolerance_ms": 200,
            "offset_tolerance_ratio": 0.2,
        },
        device="cuda",
    )

    results = evaluator.evaluate_all(
        source_audio=["./src_wavs/sample_001.wav"],
        target_audio=["./tgt_wavs/sample_001.wav"],
        # Source-side event spans should come from the dataset's explicit annotations.
        source_event_annotations=[
            [
                {"label": "laugh", "start_ms": 1200, "end_ms": 1850},
                {"label": "cough", "start_ms": 4200, "end_ms": 4550},
            ]
        ],
        # Map detector labels into the canonical label set used by the source annotations.
        event_label_mapping={
            "Laughter": "laugh",
            "Giggle": "laugh",
            "Cough": "cough",
        },
    )

    print(results)


if __name__ == "__main__":
    main()
