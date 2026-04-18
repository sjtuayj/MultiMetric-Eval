# MultiMetric-Eval

[![PyPI version](https://badge.fury.io/py/multimetriceval.svg)](https://pypi.org/project/multimetriceval/0.8.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MultiMetric-Eval 是一个聚焦于翻译与语音翻译的评测工具包。它提供统一的方式来评估文本翻译质量、语音输出质量、保真相关属性，以及流式延迟表现。

## 可用于哪些任务

这个项目最适合以下方向：

- MT 或 S2TT 的文本侧评测，支持 `BLEU`、`chrF++`、`COMET`、`BLEURT`
- S2ST 的综合评测，可组合文本质量、语音质量、说话人相似度与延迟指标
- 使用自定义 agent 的流式或同声传译延迟评测
- 面向语音翻译输出的保真分析，包括说话人、情感与副语言相似度

## 能力边界

MultiMetric-Eval 是评测工具，不是训练或推理框架。

当你已经有模型输出，并希望用统一方式进行打分时，它会比较合适。

它并不是为以下目标设计的：

- 通用 ASR 工具包
- 通用 TTS 工具包
- 模型服务框架
- 替代其他非翻译语音方向的专用工具

## 核心模块

| 模块 | 主要用途 | 常见指标 |
| :--- | :--- | :--- |
| `TranslationEvaluator` | 文本侧翻译质量评测 | `sacreBLEU`, `chrF++`, `COMET`, `BLEURT` |
| `SpeechQualityEvaluator` | 语音自然度与文本-语音一致性 | `UTMOS`, `WER_Consistency`, `CER_Consistency` |
| `SpeakerSimilarityEvaluator` | 说话人保真 | `wavlm_similarity`, `resemblyzer_similarity` |
| `EmotionEvaluator` | 情感保真或情感分类准确率 | `Emotion2Vec_Cosine_Similarity`, `Audio_Emotion_Accuracy` |
| `ParalinguisticEvaluator` | 非言语与副语言相似度 | `Paralinguistic_Fidelity_Cosine` |
| `LatencyEvaluator` | 流式 / 同传延迟评测 | `StartOffset`, `ATD`, `CustomATD`, `RTF` |

## 安装

基础安装：

```bash
pip install multimetriceval
```

可选依赖：

```bash
pip install "multimetriceval[comet]"
pip install "multimetriceval[whisper]"
pip install "multimetriceval[emotion]"
pip install "multimetriceval[paralinguistics]"
pip install "multimetriceval[all]"
```

如果你需要 BLEURT：

```bash
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
```

## 导入方式

PyPI 包名：

```python
multimetriceval
```

Python 导入名：

```python
multimetric_eval
```

示例：

```python
from multimetric_eval import TranslationEvaluator, SpeechQualityEvaluator
```

## 快速开始

### 文本翻译评测

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

### 语音质量评测

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

## 示例

示例统一放到了 `examples/` 目录。

### Python 示例

- `examples/python/translation_eval.py`
- `examples/python/speech_quality_eval.py`
- `examples/python/speaker_similarity_eval.py`
- `examples/python/emotion_eval.py`
- `examples/python/paralinguistic_eval.py`
- `examples/python/latency_eval.py`

### Bash 示例

- `examples/bash/install_extras.sh`
- `examples/bash/run_latency_cli.sh`

### 完整评测流程脚本

更完整的端到端评测脚本可以看 `test/` 目录：

- `test/run_full_eval_seamless.py`
- `test/run_full_eval_vallex.py`
- `test/run_full_eval_simulmega.py`
- `test/run_full_eval_cascade.py`

## 输入约定

常见文本输入支持：

- Python `List[str]`
- 每行一个样本的 `.txt` 文件
- `.json` 文件

常见音频输入支持：

- 文件夹路径
- Python `List[str]`
- `.txt` 文件
- `.json` 文件

## 说明

- 对于 `zh` / `ja` / `ko`，工具包在文本侧评测中使用了针对 CJK 的处理逻辑。
- `SpeechQualityEvaluator` 在 `zh` / `ja` / `ko` 上返回 `CER_Consistency`，在多数其他语言上返回 `WER_Consistency`。
- `ParalinguisticEvaluator` 当前只返回 `Paralinguistic_Fidelity_Cosine`，它是基于 embedding 的源音频与目标音频连续相似度指标。
- 某些模块依赖可选安装项，或者在离线环境中需要指定本地模型路径。

## License

MIT License
