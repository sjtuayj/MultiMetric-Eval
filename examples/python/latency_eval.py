from multimetric_eval import GenericAgent, LatencyEvaluator, ReadAction, WriteAction


class WaitUntilEndAgent(GenericAgent):
    def policy(self, states=None):
        states = states or self.states

        if not states.source_finished:
            return ReadAction()

        if not states.target_finished:
            return WriteAction("hello world", finished=True)

        return ReadAction()


def main():
    agent = WaitUntilEndAgent()
    evaluator = LatencyEvaluator(agent, segment_size=20)

    evaluator.run(
        source_files=["./data/a.wav", "./data/b.wav"],
        ref_files=["你好", "世界"],
        task="s2t",
        output_dir="./latency_output",
        visualize=False,
    )

    scores = evaluator.compute_latency(
        computation_aware=True,
        output_dir="./latency_output",
        show_all_metrics=False,
    )

    print(scores)


if __name__ == "__main__":
    main()
