from multimetric_eval import evaluate_paralinguistic_dataset, load_paralinguistic_samples


def main():
    manifest_path = "./prepared/synparaspeech_manifest.json"
    samples = load_paralinguistic_samples(manifest_path)
    source_audio_paths = [sample.source_audio for sample in samples]

    scores, diagnostics = evaluate_paralinguistic_dataset(
        target_audio=source_audio_paths,
        samples=samples,
        evaluator_kwargs={
            "use_continuous_fidelity": True,
            "use_discrete_event_f1": True,
            "clap_model_path": None,
            "discrete_event_config": {
                "clap_label_score_threshold": 0.2,
                "clap_label_fallback_top1": False,
            },
        },
        return_diagnostics=True,
    )

    print(scores)
    print(diagnostics["discrete_event_metrics"]["relaxed"]["micro"])


if __name__ == "__main__":
    main()
