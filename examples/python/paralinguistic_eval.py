from multimetric_eval import ParalinguisticEvaluator


def main():
    evaluator = ParalinguisticEvaluator(
        use_continuous_fidelity=True,
        use_discrete_matching=True,
        device="cuda",
    )

    results = evaluator.evaluate_all(
        source_audio="./src_wavs",
        target_audio="./tgt_wavs",
    )

    print(results)


if __name__ == "__main__":
    main()
