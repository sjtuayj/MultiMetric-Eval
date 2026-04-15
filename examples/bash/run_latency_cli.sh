#!/usr/bin/env bash

python -m multimetric_eval.latency.cli \
  --source data/source.txt \
  --target data/ref.txt \
  --output ./output \
  --task s2t \
  --agent-script my_agent.py \
  --agent-class MyAgent \
  --segment-size 20 \
  --computation-aware \
  --quality
