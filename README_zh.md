# MultiMetric-Eval

[![PyPI version](https://badge.fury.io/py/multimetriceval.svg)](https://pypi.org/project/multimetriceval/0.8.1/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MultiMetric-Eval 是一个面向机器翻译与语音翻译的评测工具包。它提供统一接口来评估文本翻译质量、语音输出质量、保真相关属性，以及流式延迟表现。

## 适用场景

- MT 或 S2TT 的文本侧评测，支持 `BLEU`、`chrF++`、`COMET`、`BLEURT`
- S2ST 的综合评测，可组合文本质量、语音质量、说话人相似度与延迟指标
- 使用自定义 agent 的流式或同传延迟评测
- 面向语音翻译输出的保真分析，包括说话人、情感与副语言相似度

## 能力边界

MultiMetric-Eval 是评测工具，不是训练或推理框架。

当你已经有模型输出，并希望以统一方式打分时，它比较合适。

它不用于：

- 通用 ASR 工具链
- 通用 TTS 工具链
- 模型服务框架
- 替代其他非翻译语音任务的专用工具

## 核心模块

| 模块 | 主要用途 | 常见指标 |
| :--- | :--- | :--- |
| `TranslationEvaluator` | 文本侧翻译质量评测 | `sacreBLEU`, `chrF++`, `COMET`, `BLEURT` |
| `SpeechQualityEvaluator` | 语音自然度与文本-语音一致性 | `UTMOS`, `WER_Consistency`, `CER_Consistency` |
| `SpeakerSimilarityEvaluator` | 说话人保真 | `wavlm_similarity`, `resemblyzer_similarity` |
| `EmotionEvaluator` | 情感保真或情感分类准确率 | `Emotion2Vec_Cosine_Similarity`, `Audio_Emotion_Accuracy` |
| `ParalinguisticEvaluator` | 非言语与副语言相似度 | `Paralinguistic_Fidelity_Cosine`, `Discrete_Acoustic_Event_F1_Strict`, `Discrete_Acoustic_Event_F1_Relaxed` |
| `LatencyEvaluator` | 流式 / 同传延迟评测 | `StartOffset`, `ATD`, `CustomATD`, `RTF`, `Model_Generate_RTF` |

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

### 延迟评测

```python
from multimetric_eval import GenericAgent, LatencyEvaluator, ReadAction, WriteAction


class WaitUntilEndAgent(GenericAgent):
    def policy(self, states=None):
        states = states or self.states

        if not states.source_finished:
            return ReadAction()

        if not states.target_finished:
            prediction = "hello world"
            self.record_model_inference_time(0.12)
            return WriteAction(prediction, finished=True)

        return ReadAction()


agent = WaitUntilEndAgent()
evaluator = LatencyEvaluator(agent, segment_size=20)
```

延迟输出现在区分两类 RTF：

- `Real_Time_Factor_(RTF)`：系统级 RTF，包含 agent policy、预处理、后处理以及模型推理周边开销。
- `Model_Generate_RTF`：模型级 RTF。只有 agent 显式调用 `record_model_inference_time(...)`，或在 `Segment.config["model_inference_time"]` 中提供模型推理时间时才会输出。

### 副语言评测

```python
from multimetric_eval import ParalinguisticEvaluator

evaluator = ParalinguisticEvaluator(
    use_continuous_fidelity=True,
    use_discrete_event_f1=True,
    discrete_event_config={
        "detector_backend": "panns",
        "score_threshold": 0.3,
    },
    device="cuda",
)

results = evaluator.evaluate_all(
    source_audio=["./src_wavs/sample_001.wav"],
    target_audio=["./tgt_wavs/sample_001.wav"],
    source_event_annotations=[
        [
            {"label": "laugh", "start_ms": 1200, "end_ms": 1850},
            {"label": "cough", "start_ms": 4200, "end_ms": 4550},
        ]
    ],
    event_label_mapping={
        "Laughter": "laugh",
        "Giggle": "laugh",
        "Cough": "cough",
    },
)

print(results)
```

## 示例

示例统一放在 `examples/` 目录。

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

更完整的端到端评测脚本可见 `test/`：

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

- 对 `zh` / `ja` / `ko`，工具包在文本侧评测中使用 CJK 友好的处理逻辑。
- `SpeechQualityEvaluator` 在 `zh` / `ja` / `ko` 上返回 `CER_Consistency`，在大多数其他语言上返回 `WER_Consistency`。
- `ParalinguisticEvaluator` 通过 CLAP 输出 `Paralinguistic_Fidelity_Cosine`，同时也支持输出离散事件保留指标 `Discrete_Acoustic_Event_F1_Strict` 与 `Discrete_Acoustic_Event_F1_Relaxed`。
- 内置的离散事件检测器当前使用 PANNs 后端，并依赖 `paralinguistics` extra。
- 对离散事件 F1，源端事件标签应当是规范化后的目标标签体系；`event_label_mapping` 作用在目标端检测器输出上，用于适配不同数据集或标签体系。
- 当某条样本在源端和目标端都没有事件时，该样本会在离散事件 F1 聚合时被跳过。
- 在 S2S latency 中，如果模型有原生 transcript，会优先使用原生 transcript 做对齐；如果模型只有音频输出，可以开启 ASR fallback 生成对齐文本。
- 做 S2S 强制对齐时，应显式传入目标语言对应的 MFA `alignment_acoustic_model` 与 `alignment_dictionary_model`；默认值是英文模型。
- 某些模块依赖可选安装项，或者在离线环境中需要指定本地模型路径。

## License

MIT License
