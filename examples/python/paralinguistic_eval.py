from multimetric_eval import ParalinguisticEvaluator


def strict_event_demo():
    evaluator = ParalinguisticEvaluator(
        use_continuous_fidelity=True,
        use_discrete_event_f1=True,
        discrete_event_config={
            "detector_model_path": "./model/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
            "score_threshold": 0.05,
            "clap_label_score_threshold": 0.2,
            "clap_label_fallback_top1": False,
            "onset_tolerance_ms": 200,
            "offset_tolerance_ms": 200,
            "offset_tolerance_ratio": 0.2,
            "allowed_labels": ["laughter", "coughing"],
        },
        device="cuda",
    )

    results, diagnostics = evaluator.evaluate_all(
        source_audio=["./src_wavs/sample_001.wav"],
        target_audio=["./tgt_wavs/sample_001.wav"],
        source_event_annotations=[
            [
                {"label": "laughter", "start_ms": 1200, "end_ms": 1850},
                {"label": "coughing", "start_ms": 4200, "end_ms": 4550},
            ]
        ],
        event_label_mapping={
            "Laughter": "laughter",
            "Giggle": "laughter",
            "Cough": "coughing",
        },
        return_diagnostics=True,
    )

    print("Strict event demo")
    print(results)
    print(diagnostics["discrete_event_metrics"]["prediction_source"])


def relaxed_label_demo():
    evaluator = ParalinguisticEvaluator(
        use_continuous_fidelity=True,
        use_discrete_event_f1=True,
        discrete_event_config={
            "clap_label_score_threshold": 0.2,
            "clap_label_fallback_top1": False,
        },
        device="cuda",
    )

    results, diagnostics = evaluator.evaluate_all(
        source_audio=["./src_wavs/sample_002.wav"],
        target_audio=["./tgt_wavs/sample_002.wav"],
        source_utterance_annotations=[["throat clearing"]],
        return_diagnostics=True,
    )

    print("Relaxed label demo")
    print(results)
    print(diagnostics["discrete_event_metrics"]["prediction_source"])


def main():
    strict_event_demo()
    relaxed_label_demo()


if __name__ == "__main__":
    main()
