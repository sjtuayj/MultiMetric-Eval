# MultiMetric-Eval

English | [中文](./README_zh.md)

[![PyPI version](https://badge.fury.io/py/multimetriceval.svg)](https://pypi.org/project/multimetriceval/0.8.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MultiMetric-Eval is an evaluation toolkit centered on translation and speech translation. It provides a unified way to score text translation quality, speech output quality, preservation-related properties, and streaming latency.

## What It Can Be Used For

This project is best suited for these directions:

- MT or S2TT text-side evaluation with `BLEU`, `chrF++`, `COMET`, and `BLEURT`
- S2ST evaluation by combining text quality, speech quality, speaker similarity, and latency
- Streaming or simultaneous speech translation latency evaluation with a custom agent
- Preservation analysis for speech translation outputs, including speaker similarity, emotion, and paralinguistic similarity

## Capability Boundary

MultiMetric-Eval is an evaluator, not a model training or inference framework.

It is a good fit when you already have model outputs and want to score them in a consistent way.

It is not designed to be:

- a general-purpose ASR toolkit
- a general-purpose TTS toolkit
- a model serving framework
- a replacement for task-specific toolkits in unrelated speech domains

## Core Modules

| Module | Main Use | Typical Metrics |
| :--- | :--- | :--- |
| `TranslationEvaluator` | Text-side translation quality | `sacreBLEU`, `chrF++`, `COMET`, `BLEURT` |
| `SpeechQualityEvaluator` | Naturalness and text-speech consistency | `UTMOS`, `WER_Consistency`, `CER_Consistency` |
| `SpeakerSimilarityEvaluator` | Speaker preservation | `wavlm_similarity`, `resemblyzer_similarity` |
| `EmotionEvaluator` | Emotion preservation or classification accuracy | `Emotion2Vec_Cosine_Similarity`, `Audio_Emotion_Accuracy` |
| `ParalinguisticEvaluator` | Non-verbal and paralinguistic similarity | `Paralinguistic_Fidelity_Cosine` |
| `LatencyEvaluator` | Streaming / simultaneous translation latency | `StartOffset`, `ATD`, `CustomATD`, `RTF` |

## Installation

Basic install:

```bash
pip install multimetriceval
```

Optional extras:

```bash
pip install "multimetriceval[comet]"
pip install "multimetriceval[whisper]"
pip install "multimetriceval[emotion]"
pip install "multimetriceval[paralinguistics]"
pip install "multimetriceval[all]"
```

If you need BLEURT:

```bash
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
```

## Import

PyPI package name:

```python
multimetriceval
```

Python import name:

```python
multimetric_eval
```

Example:

```python
from multimetric_eval import TranslationEvaluator, SpeechQualityEvaluator
```

## Quick Start

### Text Translation

```python
from multimetric_eval import TranslationEvaluator

evaluator = TranslationEvaluator(
    use_bleu=True,
    use_chrf=True,
    use_comet=False,
    use_bleurt=False,
    device="cuda",
)

results = evaluator.evaluate_all(
    reference=["我喜欢看电影。"],
    target_text=["我喜欢看电影。"],
    source=["I like watching movies."],
    target_lang="zh",
)

print(results)
```

### Speech Quality

```python
from multimetric_eval import SpeechQualityEvaluator

evaluator = SpeechQualityEvaluator(
    use_wer=True,
    use_utmos=True,
    whisper_model="medium",
    device="cuda",
)

results = evaluator.evaluate_all(
    target_audio="./generated_wavs",
    target_text=["你好世界", "这是一个测试"],
    target_lang="zh",
)

print(results)
```

## Examples

Examples have been moved into the `examples/` directory.

### Python Examples

- `examples/python/translation_eval.py`
- `examples/python/speech_quality_eval.py`
- `examples/python/speaker_similarity_eval.py`
- `examples/python/emotion_eval.py`
- `examples/python/paralinguistic_eval.py`
- `examples/python/latency_eval.py`

### Bash Examples

- `examples/bash/install_extras.sh`
- `examples/bash/run_latency_cli.sh`

### Full Evaluation Pipelines

For larger end-to-end evaluation scripts, see `test/`:

- `test/run_full_eval_seamless.py`
- `test/run_full_eval_vallex.py`
- `test/run_full_eval_simulmega.py`
- `test/run_full_eval_cascade.py`

## Input Conventions

Common text inputs support:

- Python `List[str]`
- `.txt` files with one sample per line
- `.json` files

Common audio inputs support:

- folder path
- Python `List[str]`
- `.txt` files
- `.json` files

## Notes

- For `zh` / `ja` / `ko`, the toolkit uses CJK-aware handling for text-side evaluation.
- `SpeechQualityEvaluator` returns `CER_Consistency` for `zh` / `ja` / `ko`, and `WER_Consistency` for most other languages.
- `ParalinguisticEvaluator` currently reports only `Paralinguistic_Fidelity_Cosine`, an embedding-based continuous similarity metric between source and target audio.
- Some modules rely on optional dependencies or local model paths in offline environments.

## License

MIT License
