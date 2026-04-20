# MultiMetric-Eval

[English](./README.md) | 中文

[![PyPI version](https://badge.fury.io/py/multimetriceval.svg)](https://pypi.org/project/multimetriceval/0.8.3/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MultiMetric-Eval 是一个面向机器翻译和语音翻译的评测工具包，提供统一的文本质量、语音质量、保留性指标和流式延迟评测接口。

## 适用方向

- MT 或 S2TT 的文本侧评测：`BLEU`、`chrF++`、`COMET`、`BLEURT`
- S2ST 的综合评测：文本质量、语音质量、说话人保持、副语言保持、延迟
- 流式或同传系统的延迟评测
- 对语音翻译输出做 preservation analysis，例如说话人、情感、副语言事件

## 核心模块

| 模块 | 主要用途 | 典型指标 |
| :--- | :--- | :--- |
| `TranslationEvaluator` | 文本翻译质量 | `sacreBLEU`, `chrF++`, `COMET`, `BLEURT` |
| `SpeechQualityEvaluator` | 自然度与语音文本一致性 | `UTMOS`, `WER_Consistency`, `CER_Consistency` |
| `SpeakerSimilarityEvaluator` | 说话人保持 | `wavlm_similarity`, `resemblyzer_similarity` |
| `EmotionEvaluator` | 情感保持或分类准确率 | `Emotion2Vec_Cosine_Similarity`, `Audio_Emotion_Accuracy` |
| `ParalinguisticEvaluator` | 副语言与非言语声学事件保持 | `Paralinguistic_Fidelity_Cosine`, `Acoustic_Event_Preservation_Rate`, `Acoustic_Event_Preservation_Macro_F1`, `Acoustic_Event_Preservation_Macro_Recall` |
| `LatencyEvaluator` | 流式 / 同传延迟 | `StartOffset`, `ATD`, `CustomATD`, `RTF`, `Model_Generate_RTF` |

## 安装

基础安装：

```bash
pip install multimetriceval
```

可选 extras：

```bash
pip install "multimetriceval[comet]"
pip install "multimetriceval[whisper]"
pip install "multimetriceval[speech_quality]"
pip install "multimetriceval[emotion]"
pip install "multimetriceval[paralinguistics]"
pip install "multimetriceval[all]"
```

如果需要 BLEURT：

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

例如：

```python
from multimetric_eval import TranslationEvaluator, SpeechQualityEvaluator
```

## 快速开始

示例脚本位于 `examples/`：

- `examples/python/translation_eval.py`
- `examples/python/speech_quality_eval.py`
- `examples/python/speaker_similarity_eval.py`
- `examples/python/emotion_eval.py`
- `examples/python/paralinguistic_eval.py`
- `examples/python/paralinguistic_identity_baseline.py`
- `examples/python/latency_eval.py`
- `examples/bash/install_extras.sh`
- `examples/bash/run_latency_cli.sh`


## 输入约定

通用文本输入支持：

- Python `List[str]`
- 每行一个样本的 `.txt`
- `.json`

通用音频输入支持：

- 文件夹路径
- Python `List[str]`
- `.txt`
- `.json`

## 说明

- `zh` / `ja` / `ko` 默认采用适合 CJK 的文本处理方式。
- `SpeechQualityEvaluator` 在 `zh` / `ja` / `ko` 上返回 `CER_Consistency`，其他多数语言返回 `WER_Consistency`。
- `ParalinguisticEvaluator` 保留 `Paralinguistic_Fidelity_Cosine`，这是基于 CLAP 的连续音频相似度指标。
- 离散副语言指标采用“单句单标签事件保持”定义；如果提供源端金标签，则输出：
  - `Acoustic_Event_Preservation_Rate`
  - `Acoustic_Event_Preservation_Macro_F1`
  - `Acoustic_Event_Preservation_Macro_Recall`
- 当前离散指标不考虑时间戳。它评估的是目标语音里是否保留了该事件，而不是事件是否出现在同一时间位置。
- 如果没有源端金标签，也可以运行 prediction-only 模式，此时输出：
  - `Predicted_Event_Consistency_Rate`
  - `Predicted_Event_Consistency_Macro_F1`
  - `Predicted_Event_Consistency_Macro_Recall`
- 默认离散预测器是基于 CLAP 的闭集分类器，候选集合由 `candidate_labels` 决定。
- 你可以传入自定义 predictor，只要它实现 `predict(audio_paths, candidate_labels)` 接口即可，无需修改核心代码。
- 数据集特定的标签映射不放进核心包。请在调用时通过 `candidate_labels` 和 `label_normalizer` 适配你的数据集。
- `clap_model_path` 同时支持 Hugging Face repo id、本地模型目录和本地 snapshot 路径，适合离线环境。
- S2S 延迟评测默认优先使用模型原生 transcript；纯语音模型可选 ASR fallback。
- 某些模块依赖可选依赖包或本地模型路径，离线环境下请自行准备。

## License

MIT License
