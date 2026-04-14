# 📊 MultiMetric-Eval

**全能型评测工具箱**：一套工具，同时满足 **机器翻译 (MT)**、**语音识别 (ASR)**、**语音合成 (TTS)**、**同声传译 (SimulST)** 与 **变声 (VC)** 的评测需求。

当前源码中的核心能力分为 6 个评测板块：

1. `TranslationEvaluator`：文本翻译质量评测
2. `SpeechQualityEvaluator`：语音自然度与文本一致性评测
3. `SpeakerSimilarityEvaluator`：说话人相似度评测
4. `EmotionEvaluator`：情感保真与情感识别评测
5. `ParalinguisticEvaluator`：副语言/非言语事件保留评测
6. `LatencyEvaluator`：流式同传时延评测

[![PyPI version](https://badge.fury.io/py/multimetriceval.svg)](https://pypi.org/project/multimetriceval/0.7.1/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 适用范围与核心能力 (Capabilities)

本工具不仅仅是一个计算器，它是一个**多模态 (Multimodal) 质量评测框架**。它不负责翻译，而是负责对 **“任意翻译/生成结果”** 进行标准化打分。

### 1. 支持的任务方向 (Supported Tasks)


MultiMetric-Eval 适合以下任务的自动评测：

| 任务类型 | 输入 -> 输出 | 推荐模块 | 典型指标 |
| :--- | :--- | :--- | :--- |
| 机器翻译（MT） | 文本 -> 文本 | `TranslationEvaluator` | BLEU, chrF++, COMET, BLEURT |
| 语音识别后评估 | 音频 -> 文本 | `SpeechQualityEvaluator` | WER / CER |
| 语音合成（TTS） | 文本 -> 音频 | `SpeechQualityEvaluator` | UTMOS, WER/CER Consistency |
| 语音翻译（S2S） | 音频 -> 音频 | `SpeechQualityEvaluator` + `TranslationEvaluator` + `LatencyEvaluator` | UTMOS, CER, Delay Metrics |
| 变声（VC） | 音频 -> 音频 | `SpeakerSimilarityEvaluator` + `SpeechQualityEvaluator` | Speaker Similarity, UTMOS |
| 情感保真 | 音频 -> 音频 | `EmotionEvaluator` | Emotion2Vec Cosine Similarity |
| 情感识别准确率 | 音频 -> 标签 | `EmotionEvaluator` | Audio Emotion Accuracy |
| 副语言事件保留 | 音频 -> 音频 | `ParalinguisticEvaluator` | Paralinguistic Fidelity, Event Retention F1 |
| 同传/流式系统 | 流式音频 -> 文本/音频 | `LatencyEvaluator` | StartOffset, ATD, CustomATD, RTF |


### 2. 多语言支持 (Language Support)

本工具针对不同语系进行了**深度优化**，解决了传统评测脚本在 CJK (中日韩) 语言上分数失真的痛点。

*   **✅ 第一梯队：深度优化 (CJK)**
    *   **语言**: **中文 (zh)**, **日语 (ja)**, **韩语 (ko)**
    *   **特性**: 内置智能路由，自动调用 **Jieba / MeCab** 进行特定分词。
    *   **优势**: 彻底解决 SacreBLEU 因无空格导致 BLEU=0 的问题；自动切换为 CER (字符错误率)。

*   **✅ 第二梯队：标准支持 (印欧语系)**
    *   **语言**: **英语 (en)**, **德语 (de)**, **法语 (fr)**, **西班牙语 (es)** 等。
    *   **特性**: 使用国际标准的 `13a` 分词器，与 WMT 评测标准对齐。

*   **✅ 第三梯队：广泛支持 (低资源语言)**
    *   **语言**: 泰语、阿拉伯语、越南语等 100+ 种语言。
    *   **特性**: 依托 **COMET** (语义模型) 和 **Whisper** (ASR模型) 的强大能力，支持绝大多数互联网语言的语义与语音评测。

---

## 🧩 模块总览

### 1. `TranslationEvaluator`
用于评测**文本翻译结果**，输入是参考文本与模型输出文本，可选源文本。

支持指标：

- `sacreBLEU`
- `chrF++`
- `COMET`
- `BLEURT`

适用场景：

- 机器翻译（MT）
- S2T/S2S 任务中，先把目标语音转写成文本后再做文本质量评测

---

### 2. `SpeechQualityEvaluator`
用于评测**生成音频本身的质量与文本一致性**。

支持指标：

- `UTMOS`：语音自然度/主观质量代理指标
- `WER_Consistency` 或 `CER_Consistency`：将生成音频转写后，与给定目标文本比较

适用场景：

- TTS
- S2S
- VC 后的可懂度/一致性检测

---

### 3. `SpeakerSimilarityEvaluator`
用于评测**参考音频与合成音频的说话人相似度**。

支持模型：

- `wavlm`
- `resemblyzer`
- `both`

输出指标：

- `wavlm_similarity`
- `resemblyzer_similarity`
- `average_wavlm_similarity`
- `average_resemblyzer_similarity`

适用场景：

- 变声（VC）
- 零样本 TTS
- 语音克隆

---

### 4. `EmotionEvaluator`
用于评测**情感是否被保留**，或者**目标音频的情感分类准确率**。

支持两类评测：

- 情感保真度：`Emotion2Vec_Cosine_Similarity`
- 情感识别准确率：`Audio_Emotion_Accuracy`

适用场景：

- 跨语种语音翻译中的情感保留
- 情感 TTS / 情感 VC
- 语音情感识别结果校验

---

### 5. `ParalinguisticEvaluator`
用于评测**副语言与非言语事件是否被保留**。

支持两类指标：

- `Paralinguistic_Fidelity_Cosine`
- `Event_Retention_F1`

默认关注事件包括：

- `laughter`
- `coughing`
- `sighing`
- `breathing heavily`
- `throat clearing`

适用场景：

- S2S 翻译中笑声、咳嗽、叹气等是否保留
- 非语言声学事件是否在生成端被错误丢失或错误添加

---

### 6. `LatencyEvaluator`
用于评测**流式/同传系统的延迟表现**。

支持任务：

- `s2t`
- `s2s`

支持指标：

- `StartOffset`
- `ATD`
- `CustomATD`
- `RTF`

在有对齐支持时还可得到：

- `StartOffset_SpeechAlign`
- `ATD_SpeechAlign`
- `CustomATD_SpeechAlign`

同时支持：

- computation-aware 统计
- 自定义 Agent
- 串联 AgentPipeline
- CLI 运行

---

### 7. 数据集模块
工具包提供内置数据集和本地 JSON 数据构建能力。

可用接口：

- `list_datasets()`
- `get_dataset_info(name)`
- `load_dataset(name)`
- `create_dataset_from_json(json_path)`

当前内置数据集包括：

- `zh-en-littleprince`
- `materials2`
- `RAVDESS`
- `flores200_f5_audio`

---

## 🚀 安装

### 基础安装

```bash
pip install multimetriceval
```

### 按需安装可选依赖

```bash
# COMET
pip install "multimetriceval[comet]"

# Whisper
pip install "multimetriceval[whisper]"

# EmotionEvaluator 依赖
pip install "multimetriceval[emotion]"

# ParalinguisticEvaluator 依赖
pip install "multimetriceval[paralinguistics]"

# 全部可选依赖
pip install "multimetriceval[all]"
```

### BLEURT（可选）
如需使用 BLEURT，需要额外安装：

```bash
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
```

---
## 📦 导入方式

注意：

- PyPI 包名是 `multimetriceval`
- Python 导入名是 `multimetric_eval`

正确导入方式：

```python
from multimetric_eval import (
    TranslationEvaluator,
    SpeechQualityEvaluator,
    SpeakerSimilarityEvaluator,
    EmotionEvaluator,
    ParalinguisticEvaluator,
    LatencyEvaluator,
    Dataset,
    load_dataset,
    list_datasets,
    get_dataset_info,
    create_dataset_from_json,
)
```

延迟模块也可以单独导入：

```python
from multimetric_eval.latency import (
    GenericAgent,
    AgentPipeline,
    ReadAction,
    WriteAction,
    LatencyEvaluator,
)
```

---

## ⚡ 快速开始

### 1. 文本翻译评测

```python
from multimetric_eval import TranslationEvaluator

evaluator = TranslationEvaluator(
    use_bleu=True,
    use_chrf=True,
    use_comet=True,
    use_bleurt=False,
    device="cuda"
)

results = evaluator.evaluate_all(
    reference=["猫坐在垫子上。"],
    target_text=["猫坐在垫子上。"],
    source=["The cat sits on the mat."],
    target_lang="zh"
)

print(results)
```

示例输出：

```python
{
    "sacreBLEU": 100.0,
    "chrF++": 100.0,
    "COMET": 0.99
}
```

---

### 2. 语音质量评测

```python
from multimetric_eval import SpeechQualityEvaluator

evaluator = SpeechQualityEvaluator(
    use_wer=True,
    use_utmos=True,
    whisper_model="medium",
    device="cuda"
)

results = evaluator.evaluate_all(
    target_audio="./generated_wavs",
    target_text=["你好世界", "这是一个测试"],
    target_lang="zh"
)

print(results)
```

可能输出：

```python
{
    "UTMOS": 3.98,
    "CER_Consistency": 0.07
}
```

---

### 3. 说话人相似度评测

```python
from multimetric_eval import SpeakerSimilarityEvaluator

evaluator = SpeakerSimilarityEvaluator(
    model_type="both",
    device="cuda"
)

results = evaluator.evaluate_batch(
    ref_wav_paths=["./ref/1.wav", "./ref/2.wav"],
    synth_wav_paths=["./gen/1.wav", "./gen/2.wav"]
)

print(results["average_wavlm_similarity"])
print(results["average_resemblyzer_similarity"])
```

---

### 4. 情感保真评测

```python
from multimetric_eval import EmotionEvaluator

evaluator = EmotionEvaluator(
    e2v_model_path="iic/emotion2vec_plus_large",
    device="cuda"
)

results = evaluator.evaluate_all(
    source_audio="./src_wavs",
    target_audio="./tgt_wavs"
)

print(results)
```

可能输出：

```python
{
    "Emotion2Vec_Cosine_Similarity": 0.81
}
```

---

### 5. 情感分类准确率评测

```python
from multimetric_eval import EmotionEvaluator

evaluator = EmotionEvaluator(device="cuda")

results = evaluator.evaluate_all(
    target_audio="./emotion_wavs",
    reference_labels=["happy", "sad", "neutral", "happy"]
)

print(results)
```

可能输出：

```python
{
    "Audio_Emotion_Accuracy": 0.75
}
```

---

### 6. 副语言事件评测

```python
from multimetric_eval import ParalinguisticEvaluator

evaluator = ParalinguisticEvaluator(
    use_continuous_fidelity=True,
    use_discrete_matching=True,
    device="cuda"
)

results = evaluator.evaluate_all(
    source_audio="./src_wavs",
    target_audio="./tgt_wavs"
)

print(results)
```

可能输出：

```python
{
    "Paralinguistic_Fidelity_Cosine": 0.42,
    "Event_Retention_F1": 0.89
}
```

---

## 📘 详细使用说明

## 一、`TranslationEvaluator`

### 接口

```python
TranslationEvaluator(
    use_bleu=True,
    use_chrf=True,
    use_comet=True,
    use_bleurt=False,
    comet_model="Unbabel/wmt22-comet-da",
    bleurt_path=None,
    bleurt_model=None,
    device=None
)
```

主入口：

```python
evaluate_all(
    reference,
    target_text,
    source=None,
    target_lang="en"
)
```

### 参数说明

- `reference`：参考译文列表，或 `.txt` / `.json` 文件路径
- `target_text`：模型输出文本列表，或 `.txt` / `.json` 文件路径
- `source`：源文本列表，或 `.txt` / `.json` 文件路径；`COMET` 需要它
- `target_lang`：目标语言，用于选择 BLEU tokenizer

### 支持的文本输入格式

#### 1. Python 列表
```python
reference = ["你好"]
target_text = ["你好"]
source = ["hello"]
```

#### 2. `.txt`
每行一条：

```text
你好
这是一个测试
```

#### 3. `.json`
支持以下几类：

```json
["你好", "这是一个测试"]
```

```json
{"target_text": ["你好", "这是一个测试"]}
```

```json
[
  {"target_text": "你好"},
  {"target_text": "这是一个测试"}
]
```

### CJK 语言说明

对于 `zh` / `ja` / `ko`，请显式传入 `target_lang`：

```python
results = evaluator.evaluate_all(
    reference=refs,
    target_text=hyps,
    source=srcs,
    target_lang="zh"
)
```

对应 tokenizer：

- `zh` -> `zh`
- `ja` -> `ja-mecab`
- `ko` -> `ko-mecab`
- 其他语言 -> `13a`

---

## 二、`SpeechQualityEvaluator`

### 接口

```python
SpeechQualityEvaluator(
    use_wer=True,
    use_utmos=True,
    whisper_model="medium",
    utmos_model_path=None,
    utmos_ckpt_path=None,
    device=None
)
```

主入口：

```python
evaluate_all(
    target_audio,
    target_text=None,
    target_lang="en"
)
```

### 参数说明

- `target_audio`：生成音频列表，或音频文件夹路径
- `target_text`：模型同步生成的目标文本，用于 WER/CER 一致性评测
- `target_lang`：用于决定返回 `WER_Consistency` 还是 `CER_Consistency`

### 说明

- 如果 `target_text` 缺失，则只计算 `UTMOS`
- 对 `zh` / `ja` / `ko`，会自动按字符级评测，返回 `CER_Consistency`
- 对其他语言，返回 `WER_Consistency`

### 示例

```python
from multimetric_eval import SpeechQualityEvaluator

evaluator = SpeechQualityEvaluator(
    use_wer=True,
    use_utmos=True,
    device="cuda"
)

results = evaluator.evaluate_all(
    target_audio="./tts_outputs",
    target_text="./tts_outputs.txt",
    target_lang="en"
)
```

---

## 三、`SpeakerSimilarityEvaluator`

### 接口

```python
SpeakerSimilarityEvaluator(
    model_type="wavlm",
    device=None,
    wavlm_model_path="microsoft/wavlm-base-plus-sv",
    resemblyzer_weights_path="pretrained.pt"
)
```

### 单对评测

```python
from multimetric_eval import SpeakerSimilarityEvaluator

evaluator = SpeakerSimilarityEvaluator(model_type="both", device="cuda")

result = evaluator.evaluate(
    ref_wav_path="./ref.wav",
    synth_wav_path="./synth.wav"
)

print(result)
```

### 批量评测

```python
results = evaluator.evaluate_batch(
    ref_wav_paths=["./ref1.wav", "./ref2.wav"],
    synth_wav_paths=["./gen1.wav", "./gen2.wav"]
)
```

### 输出说明

- `wavlm_similarity`
- `resemblyzer_similarity`
- `average_wavlm_similarity`
- `average_resemblyzer_similarity`

---

## 四、`EmotionEvaluator`

### 接口

```python
EmotionEvaluator(
    e2v_model_path=None,
    custom_label_map=None,
    device=None
)
```

主入口：

```python
evaluate_all(
    source_audio=None,
    target_audio=None,
    reference_labels=None,
    verbose=True
)
```

### 两种模式

#### 1. 情感保真模式
同时传入 `source_audio` 和 `target_audio`：

```python
results = evaluator.evaluate_all(
    source_audio="./src_wavs",
    target_audio="./tgt_wavs"
)
```

输出：

- `Emotion2Vec_Cosine_Similarity`

#### 2. 情感分类准确率模式
传入 `target_audio` 和 `reference_labels`：

```python
results = evaluator.evaluate_all(
    target_audio="./tgt_wavs",
    reference_labels=["happy", "sad", "neutral"]
)
```

输出：

- `Audio_Emotion_Accuracy`

### 标签映射

```python
evaluator = EmotionEvaluator(
    custom_label_map={
        "angry": "anger",
        "hap": "happy",
        "exc": "happy"
    }
)
```

### 输入格式

`target_audio` / `source_audio` 支持：

- 文件夹路径
- 文件路径列表
- `.txt`
- `.json`

`reference_labels` 支持：

- 标签列表
- `.txt`
- `.json`

---

## 五、`ParalinguisticEvaluator`

### 接口

```python
ParalinguisticEvaluator(
    use_continuous_fidelity=True,
    use_discrete_matching=True,
    clap_model_path=None,
    device=None
)
```

主入口：

```python
evaluate_all(
    source_audio,
    target_audio,
    verbose=True
)
```

### 输出指标

- `Paralinguistic_Fidelity_Cosine`
- `Event_Retention_F1`

### 示例

```python
from multimetric_eval import ParalinguisticEvaluator

evaluator = ParalinguisticEvaluator(device="cuda")

results = evaluator.evaluate_all(
    source_audio="./src_audio",
    target_audio="./tgt_audio"
)

print(results)
```

### 说明

- 连续特征保真使用 CLAP 音频 embedding 的余弦相似度
- 离散事件保留使用候选事件集合的 F1
- 当前离散事件检测是 zero-shot 风格的伪检测逻辑，适合作为统一 benchmark 指标，但不是专门的事件检测器

---

## 六、`LatencyEvaluator`

`LatencyEvaluator` 面向流式系统评测。  
你需要实现一个 `GenericAgent`，系统会模拟分块输入、记录每一步的读写时序，并计算延迟指标。

### 延迟抽象

延迟模块提供几个基础概念：

- `ReadAction`：继续读取输入
- `WriteAction`：输出文本或音频片段
- `GenericAgent`：用户自定义流式策略
- `AgentPipeline`：多个 Agent 串联
- `SpeechToTextInstance` / `SpeechToSpeechInstance`：内部实例对象

### 最小 Agent 示例

```python
from multimetric_eval.latency import GenericAgent, ReadAction, WriteAction

class MyAgent(GenericAgent):
    def policy(self, states=None):
        states = states or self.states

        if not states.source_finished:
            return ReadAction()

        if not states.target_finished:
            return WriteAction("hello world", finished=True)

        return ReadAction()
```

### Python 中使用

```python
from multimetric_eval.latency import LatencyEvaluator

agent = MyAgent()
evaluator = LatencyEvaluator(agent, segment_size=20)

instances = evaluator.run(
    source_files=["./a.wav", "./b.wav"],
    ref_files=["你好", "世界"],
    task="s2t",
    output_dir="./latency_output"
)

scores = evaluator.compute_latency(
    computation_aware=True,
    output_dir="./latency_output",
    show_all_metrics=False
)

print(scores)
```

### CLI 使用

```bash
python -m multimetric_eval.latency.cli ^
  --source data/source.txt ^
  --target data/ref.txt ^
  --output ./output ^
  --task s2t ^
  --agent-script my_agent.py ^
  --agent-class MyAgent ^
  --segment-size 20 ^
  --computation-aware ^
  --quality
```

### CLI 参数

- `--source`：源音频路径列表文件
- `--target`：参考文本列表文件，可选
- `--output`：输出目录
- `--task`：`s2t` 或 `s2s`
- `--agent-script`：包含 Agent 类的 Python 文件
- `--agent-class`：Agent 类名
- `--segment-size`：输入切片大小，单位毫秒
- `--computation-aware`：计算 computation-aware 指标
- `--quality`：评测结束后额外做质量评估
- `--slurm`：通过 slurm 提交

### 推荐指标

默认推荐展示 4 个指标：

- `First_Audio_Delay_(ALAL_ms)`：首输出延迟
- `Overall_Translation_Delay_(ATD_ms)`：整体翻译延迟
- `End_Action_Delay_(CustomATD_ms)`：末端动作延迟
- `Real_Time_Factor_(RTF)`：实时率

如果启用 `show_all_metrics=True`，还会返回全部底层 scorer 的详细结果。

---
## 🗂️ 数据集工具

### 列出内置数据集

```python
from multimetric_eval import list_datasets

print(list_datasets())
```

### 查看数据集信息

```python
from multimetric_eval import get_dataset_info

info = get_dataset_info("zh-en-littleprince")
print(info)
```

### 加载内置数据集

```python
from multimetric_eval import load_dataset

dataset = load_dataset("zh-en-littleprince")
print(len(dataset))
print(dataset[0])
print(dataset.source_texts[:3])
print(dataset.reference_texts[:3])
```

### 从本地 JSON 创建数据集

```python
from multimetric_eval import create_dataset_from_json

dataset = create_dataset_from_json("./my_dataset.json")
print(len(dataset))
```

### `Dataset` 对象提供的属性

- `ids`
- `source_texts`
- `reference_texts`
- `audio_paths`
- `verify_audio_files()`

---

## 📂 输入格式约定

### 文本类输入
支持：

- Python `List[str]`
- `.txt`：每行一条
- `.json`

### 音频类输入
支持：

- 文件夹路径
- Python `List[str]`
- `.txt`
- `.json`

扫描文件夹时默认支持扩展名：

- `.wav`
- `.mp3`
- `.flac`

并按文件名排序。

### 标签类输入
支持：

- Python `List[str]`
- `.txt`
- `.json`

---

## 🌍 多语言支持

### 文本翻译评测
`TranslationEvaluator` 会根据 `target_lang` 自动切换 BLEU tokenizer：

- `zh` -> `zh`
- `ja` -> `ja-mecab`
- `ko` -> `ko-mecab`
- 其他 -> `13a`

### 语音一致性评测
`SpeechQualityEvaluator` 会根据 `target_lang` 自动决定：

- `zh` / `ja` / `ko` -> `CER_Consistency`
- 其他 -> `WER_Consistency`

---

## ⚙️ 离线 / 内网环境

### 1. COMET / BLEURT
建议提前把模型下载到本地缓存或指定本地路径。

### 2. Whisper
`SpeechQualityEvaluator` 使用 `openai-whisper`。  
如果是离线环境，请预先把对应模型下载到本地缓存。

### 3. UTMOS
`SpeechQualityEvaluator` 中的 UTMOS 通过 `torch.hub.load()` 加载 `tarepan/SpeechMOS`。  
如果离线使用，请提前准备：

- `SpeechMOS` 仓库本地路径
- UTMOS checkpoint 本地路径

示例：

```python
evaluator = SpeechQualityEvaluator(
    use_utmos=True,
    utmos_model_path="/path/to/SpeechMOS",
    utmos_ckpt_path="/path/to/utmos22_strong.pth",
    device="cuda"
)
```

### 4. Emotion2Vec
`EmotionEvaluator` 支持本地模型路径：

```python
evaluator = EmotionEvaluator(
    e2v_model_path="/path/to/emotion2vec_plus_large",
    device="cuda"
)
```

### 5. CLAP
`ParalinguisticEvaluator` 支持本地模型路径：

```python
evaluator = ParalinguisticEvaluator(
    clap_model_path="/path/to/local_clap_model",
    device="cuda"
)
```

---

## ⚙️ 全局配置

### GPU 设置
所有模块均支持自动检测 GPU。手动指定方式如下：

```python
# 指定第 0 号 GPU
evaluator = TranslationEvaluator(device="cuda:0")

# 强制使用 CPU
evaluator = TranslationEvaluator(device="cpu")
```

或者使用环境变量：
```bash
CUDA_VISIBLE_DEVICES=1 python my_script.py
```

### 国内网络镜像
如果下载模型超时，请在代码开头设置 HF 镜像：
```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```
---

## 📊 指标速查表

| 模块 | 指标 | 含义 |
| :--- | :--- | :--- |
| `TranslationEvaluator` | `sacreBLEU` | 传统 n-gram 翻译质量指标 |
| `TranslationEvaluator` | `chrF++` | 字符级 F-score，适合形态丰富语言 |
| `TranslationEvaluator` | `COMET` | 语义级翻译质量指标 |
| `TranslationEvaluator` | `BLEURT` | 基于预训练语言模型的质量指标 |
| `SpeechQualityEvaluator` | `UTMOS` | 语音自然度代理指标 |
| `SpeechQualityEvaluator` | `WER_Consistency` | 英文等语言的文本-语音一致性错误率 |
| `SpeechQualityEvaluator` | `CER_Consistency` | 中日韩语言的字符级一致性错误率 |
| `SpeakerSimilarityEvaluator` | `wavlm_similarity` | 基于 WavLM 的说话人相似度 |
| `SpeakerSimilarityEvaluator` | `resemblyzer_similarity` | 基于 Resemblyzer 的说话人相似度 |
| `EmotionEvaluator` | `Emotion2Vec_Cosine_Similarity` | 原音频与目标音频的情感 embedding 相似度 |
| `EmotionEvaluator` | `Audio_Emotion_Accuracy` | 目标音频的情感分类准确率 |
| `ParalinguisticEvaluator` | `Paralinguistic_Fidelity_Cosine` | 副语言整体声学特征相似度 |
| `ParalinguisticEvaluator` | `Event_Retention_F1` | 非言语事件集合的保留 F1 |
| `LatencyEvaluator` | `StartOffset` | 首次输出延迟 |
| `LatencyEvaluator` | `ATD` | 平均 token 延迟 |
| `LatencyEvaluator` | `CustomATD` | 去除目标音频物理时长后的尾部动作延迟 |
| `LatencyEvaluator` | `RTF` | 推理时间 / 源音频时长 |


---

## ❓ 常见问题 (FAQ)

**Q: 为什么中文计算出的 WER 是 1.0 (100%)？**
A: 这是因为传统 WER 算法以空格分词，中文没有空格会被当成一个词，错一个字就算全错。**本工具已修复此问题**，会自动给中文字符间插入空格，计算 CER，请放心使用。

**Q: UTMOS 加载报错 `HTTP 404` 或 `Connection Refused`？**
A: 这是因为服务器无法连接 GitHub 下载源码。请参考上文 **“离线 / 内网环境使用指南”**，手动下载 `SpeechMOS` 源码包并指定 `utmos_model_path`。

**Q: 安装时提示 `ffmpeg` not found?**
A: Whisper 依赖系统级的 ffmpeg。请安装它：
*   Ubuntu: `sudo apt install ffmpeg`
*   Windows: 下载 ffmpeg.exe 并添加到环境变量 PATH 中。

**Q: `SpeechEvaluator` 报错缺少 Whisper?**
A: 请运行 `pip install "multimetriceval[whisper]"`。如果不想用 WER 指标，可以忽略该错误，UTMOS 依然可用。

**Q: `LatencyEvaluator` 报错 `matplotlib` not found?**
A: 请运行 `pip install "multimetriceval[viz]"` 安装可视化依赖。

**Q: S2S 任务中 `ATD_SpeechAlign` 没有结果？**
A: 这通常是因为未安装 MFA 或未下载 MFA 模型。请参考模块三文档安装 `montreal-forced-aligner`。如果未安装，工具会自动跳过对齐指标，仅计算基础切片延迟。

**Q: 评测日语时提示 `ModuleNotFoundError: No module named 'MeCab'`？**
A: 评测日语 BLEU 需要依赖 MeCab。请运行 `pip install mecab-python3 ipadic` 安装所需依赖。同理，中文评测需要 `pip install jieba`。

**Q: 如何在 Python 代码中手动调用同传评测？**
```python
from multimetriceval.latency import GenericAgent, LatencyEvaluator

# 定义 Agent
class MyAgent(GenericAgent): ...

# 运行评测
agent = MyAgent()
evaluator = LatencyEvaluator(agent)
instances = evaluator.run(src_files, ref_files, task="s2t")

# 计算分数
scores = evaluator.compute_latency(computation_aware=True)
print(scores)
```

---

## 📜 License

MIT License

## 🤝 Contributing

欢迎提交 Issue 和 Pull Request！
GitHub: [https://github.com/sjtuayj/MultiMetric-Eval](https://github.com/sjtuayj/MultiMetric-Eval)