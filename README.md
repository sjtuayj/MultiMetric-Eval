# 📊 MultiMetric-Eval

**全能型评测工具箱**：一套工具，同时满足 **机器翻译 (MT)**、**语音识别 (ASR)**、**语音合成 (TTS)**、**同声传译 (SimulST)** 与 **变声 (VC)** 的评测需求。

[![PyPI version](https://badge.fury.io/py/multimetriceval.svg)](https://pypi.org/project/multimetriceval/0.7.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 适用范围与核心能力 (Capabilities)

本工具不仅仅是一个计算器，它是一个**多模态 (Multimodal) 质量评测框架**。它不负责翻译，而是负责对 **“任意翻译/生成结果”** 进行标准化打分。

### 1. 支持的任务方向 (Supported Tasks)

无论你的模型是文本还是语音，只要有参考答案 (Reference)，本工具均可评测：

| 任务类型 | 输入 -> 输出 | 关键指标 | 典型场景 |
| :--- | :--- | :--- | :--- |
| **机器翻译 (MT)** | 文本 -> 文本 | BLEU, chrF++, COMET, BLEURT | 文档翻译、聊天机器人 |
| **语音识别 (ASR)** | 语音 -> 文本 | WER / CER | 会议记录、字幕生成 |
| **语音合成 (TTS)** | 文本 -> 语音 | UTMOS (自然度), ASR-WER (可懂度) | 文本朗读、数字人 |
| **语音翻译 (S2T)** | 语音 -> 文本 | BLEU, COMET, WER | 端到端语音翻译 |
| **语音转语音 (S2S)**| 语音 -> 语音 | UTMOS, ASR-BLEU, ASR-COMET | 同声传译、变声器 |
| **跨语种情感保真度**| 音频 -> 离散/连续差距 | Fidelity (Cosine) / Accuracy | 同传、变声情感保留度测试 |
| **副语言/声学事件保留 (<ins>New!</ins>)**| 音频 -> 音频 | Paralinguistic Fidelity / Event F1 | 翻译/变声中的笑声、咳嗽等非语言声音保留度测试 |

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

## 🌟 核心功能

本工具包含四个核心评测模块：

1.  **`TranslationEvaluator` (全能评测器)**
    *   **一站式解决方案**: 同时支持 **文本翻译** 和 **语音合成/转换** 的评测。
    *   **文本指标**: BLEU, chrF++, COMET, BLEURT。
    *   **语音指标**: UTMOS (自然度), WER/CER (识别准确率)。
    *   **特点**: 智能判断输入类型（文本/语音），自动计算所有适用的指标。

2.  **`LatencyEvaluator` (同传延迟评测)**
    *   **场景**: 同声传译 (S2T / S2S)。
    *   **指标**: StartOffset, ATD (Average Token Delay)。
    *   **特点**:
        *   支持 **计算感知 (Computation Aware)** 延迟。
        *   支持 **MFA 强制对齐**，实现精确的 S2S 延迟测量。
        *   **一键集成质量评测** (自动调用 TranslationEvaluator)。

3.  **`EmotionEvaluator` (基于大模型特征的情感保真度评测器)**
    *   **场景**: S2ST (语音到语音翻译)、变声等跨文化/跨语种声音情感保留一致性评测。
    *   **指标**: 
        *   **Emotion Fidelity (Cosine Similarity)**: 基于 Emotion2Vec+ 提取的 768-d 高维嵌入特征，计算原音频和目标音频的情感保留余弦相似度。
        *   **Emotion Recognition Accuracy**: 将模型用作 Zero-shot 分类器，评测生成声音情感与参考标签的一致率。
    *   **特点**: 完全基于开源前沿声音情感大模型 `iic/emotion2vec_plus_large`，鲁棒应对多种语种发音。

4.  **`ParalinguisticEvaluator` (副语言与声学事件评测器)**  <mark>🆕 新增</mark>
    *   **场景**: 跨语种 S2ST 或变声任务中，评估模型在生成目标语音时，能否较好地**保留原音频中存在的物理发声（如叹气、咳嗽、笑声、重呼吸等环境声学事件）**。
    *   **特点**: 利用多模态 CLAP 模型的声学空间表征及零样本 (Zero-Shot) 跨模态比对能力，实现纯客观盲测。


---

## 🚀 安装

### 1. 基础安装

```bash
# 基础版 (仅支持文本指标)
pip install multimetriceval
```

### 2. 按需安装依赖

根据你需要使用的功能安装额外的库：

```bash
# [文本] 启用 COMET (语义评测)
pip install "multimetriceval[comet]"

# [语音] 启用 Whisper (ASR / WER / 语音转文本)
pip install "multimetriceval[whisper]"

# [情感] 启用 EmotionEvaluator
pip install "multimetriceval[emotion]"

# [副语言] 启用 ParalinguisticEvaluator (需包含在本地代码中调用)
pip install transformers librosa soundfile

# 安装所有功能 (推荐)
pip install "multimetriceval[all]"
```


### 3. 安装 BLEURT (可选)

如果你需要使用 BLEURT 指标（PyTorch版），需手动安装：
```bash
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
```

---

## 📖 模块一：全能评测 (`TranslationEvaluator`)

这是本库的核心类，能够处理从纯文本到纯语音，以及混合模态的所有评测需求。

### ⚡ 初始化配置 (Switches)

通过初始化参数，你可以精确控制要计算哪些指标：

```python
from multimetriceval import TranslationEvaluator

evaluator = TranslationEvaluator(
    # --- 文本指标开关 ---
    use_bleu=True,      # 默认 True
    use_chrf=True,      # 默认 True
    use_comet=True,     # 默认 True (需安装 comnet)
    use_bleurt=False,   # 默认 False (需手动安装)
    
    # --- 语音指标开关 ---
    use_wer=True,       # 计算 WER/CER (需开启 whisper)
    use_utmos=False,    # 计算自然度 UTMOS (需下载模型)
    use_whisper=False,  # 启用 ASR 转写功能
    
    # --- 模型路径 (可选) ---
    device="cuda"       # 自动检测，可手动指定
)
```
### 📂 数据输入方式详解

`evaluate_all` 方法非常灵活，支持 **内存变量 (List)** 和 **文件路径 (Path)** 混合输入。

#### 1. 文本输入 (`target_text`&`reference` & `source`)
支持直接传入 Python 列表，或传入文件路径（工具会自动读取）。
一致性要求: 无论使用哪种方式，source 和 reference 的行数/条目数必须与评测目标严格对应。

*   **格式 A: 纯文本文件 (.txt)**
    *   要求：一行一句，数量必须与 `reference` 一致。
    ```text
    Hello world
    This is a test
    ```
*   **格式 B: JSON 文件 (.json)**
    *   支持三种结构，工具会自动识别：
    1.  **字典 (推荐)**: `{"target_text": ["句1", "句2"]}` 或 `{"hypothesis": ["...", "..."]}`
    2.  **列表-字典**: `[{"target_text": "句1"}, {"target_text": "句2"}]`
    3.  **纯列表**: `["句1", "句2"]`

#### 2. 语音输入 (`target_speech`)
支持直接传入音频路径列表，或传入文件夹路径。

*   **方式 A: 文件夹路径 (推荐)**
    *   传入字符串路径：`"./audio_output/"`
    *   **重要规则**：工具会自动读取该文件夹下所有 `.wav/.mp3/.flac` 文件，并**按文件名 (A-Z) 排序**。请确保排序后的音频与 `reference` 文本顺序一一对应。
*   **方式 B: 文件路径列表**
    *   传入 List：`["/data/1.wav", "/data/2.wav"]`


---
### 🛠️ 使用场景示例

#### 场景 1: 纯文本机器翻译 (MT)
计算 BLEU, chrF++, COMET。
**注意：** 如果目标语言是中文 (`zh`) 或日语 (`ja`)，请务必指定 `target_lang`，工具会自动切换分词器 (jieba/mecab)。

```python
evaluator = TranslationEvaluator(use_comet=True)

# 示例：英日翻译 (English -> Japanese)
results = evaluator.evaluate_all(
    target_text=["猫はマットの上に座っている。"],  # 日语翻译结果
    reference=["猫はマットの上に座っています。"], # 日语参考译文
    source=["The cat sits on the mat."],       # 源文本
    target_lang="ja"                           # <--- 关键！指定目标语言为日语
)

# 工具会自动调用 'ja-mecab' 分词，计算准确的 BLEU
print(results)
# {'sacreBLEU': 45.2, 'chrF++': 62.1, 'COMET': 0.85}
```

#### 场景 2: 语音合成 (TTS) / 变声 (VC)
计算 **UTMOS (自然度)** 和 **WER/CER (准确率)**。
*注意：计算 WER 需要开启 `use_whisper=True`。*

```python
# 初始化: 开启语音相关功能
evaluator = TranslationEvaluator(
    use_utmos=True, 
    use_whisper=True, 
    use_wer=True,
    device="cuda"
)

results = evaluator.evaluate_all(
    target_speech="./generated_audio_folder/",  # 音频文件夹
    reference=["你好世界", "这是一个测试"],      # 对应的文本内容
    target_lang="zh"                            # <--- 指定中文，影响 ASR 后计算的 BLEU 分词
)

print(results)
# {
#   'UTMOS': 4.15,          # 自然度得分
#   'WER': 0.05,            # 字错误率 (中文自动转为 CER)
#   'sacreBLEU_ASR': 98.5   # ASR文本与参考文本的 BLEU (自动使用 zh 分词)
# }
```

#### 场景 3: 语音翻译 (S2T / S2S) - 双模态评测
同时评估 **“生成的文本”** 和 **“生成的语音”**。常用于端到端语音翻译系统。

```python
results = evaluator.evaluate_all(
    reference=["Ref sentence"], 
    source=["Src sentence"],
    target_text="./outputs/text_predictions.txt", # 文本结果文件
    target_speech="./outputs/audio_predictions/"  # 语音结果文件夹
)

# 返回结果将同时包含:
# 1. 文本指标 (基于 target_text): sacreBLEU, COMET
# 2. 语音指标 (基于 target_speech): UTMOS, WER, sacreBLEU_ASR, COMET_ASR
```

---
## 🌍 多语言支持：解决 SacreBLEU 分词问题

在评测 **中文 (zh)**、**日语 (ja)** 或 **韩语 (ko)** 时，传统的 BLEU 算法（默认以空格分词）会导致评分严重失真（通常接近 0 分）。

本工具内置了**智能分词路由**，你只需在 `evaluate_all` 中传入 `target_lang` 参数，工具会自动处理底层细节：

| 目标语言 | `target_lang` | 自动调用的 Tokenizer | 依赖库 |
| :--- | :--- | :--- | :--- |
| **日语** | `"ja"` | `ja-mecab` | `mecab-python3`, `ipadic` |
| **中文** | `"zh"` | `zh` | `jieba` |
| **韩语** | `"ko"` | `ko-mecab` | `mecab-ko` |
| **英语/其他** | `"en"`, `"fr"`... | `13a` (默认) | 无 |

**使用方式：**
```python
# 日语评测 (必须指定，否则 BLEU 极低)
evaluator.evaluate_all(..., target_lang="ja")

# 英语评测 (默认可不写，或写 "en")
evaluator.evaluate_all(..., target_lang="en")
```
---

## ⚡️ 语音评测核心特性

### 1. WER/CER 智能适配
本工具内置了智能分词逻辑：
*   **英文/印欧语系**：保持空格分词，计算 **WER (Word Error Rate)**。
*   **中文/日文**：自动在字符间插入空格，计算 **CER (Character Error Rate)**。
*   **混合文本**：混合处理。
> 无需用户手动分词，直接传入原始文本即可。

### 2. ASR 后文本指标联动
当 `use_whisper=True` 时，工具会计算 `sacreBLEU_ASR`。该指标同样受 `target_lang` 参数控制。例如设置 `target_lang="ja"`，不仅 WER 会按字符计算，`sacreBLEU_ASR` 也会自动使用 MeCab 分词。

### 3. 双轨制指标命名
当同时传入文本和语音时，结果字典中的 key 遵循以下规则：
*   **标准指标** (如 `sacreBLEU`, `COMET`): 基于 `target_text` 计算。
*   **ASR 指标** (如 `sacreBLEU_ASR`, `COMET_ASR`): 先用 Whisper 将 `target_speech` 转写为文本，再基于转写结果计算。

---

## ⏱️ 模块二：同传延迟评测 (`LatencyEvaluator`)

专为 **Simultaneous Translation (同声传译)** 任务设计，支持 S2T (Speech-to-Text) 和 S2S (Speech-to-Speech)。

> **核心理念**：用户只需定义 Agent（读/写策略），本工具负责模拟流式环境、计算延迟并自动调用质量评测。

### 1. 编写你的 Agent

你需要创建一个 Python 文件（例如 `my_agent.py`），继承 `GenericAgent` 并实现 `policy` 方法。

```python
# my_agent.py
from multimetriceval.latency import GenericAgent, ReadAction, WriteAction

class MyAgent(GenericAgent):
    def policy(self, states=None):
        # 策略示例：如果源端没读完，就请求更多音频 (Read)
        if not self.states.source_finished:
            return ReadAction()
        
        # 如果源端读完了，但还没输出，就输出翻译 (Write)
        if not self.states.target_finished:
            return WriteAction("Hello World", finished=True)
            
        return ReadAction()
```

### 2. 命令行一键评测

使用内置的 CLI 工具运行评测。支持同时计算 **延迟指标** 和 **质量指标**。

```bash
python -m multimetriceval.latency.cli \
    --source data/audio_list.txt \
    --target data/ref.txt \
    --agent-script my_agent.py \
    --agent-class MyAgent \
    --task s2t \
    --computation-aware \
    --quality
```

#### 参数详解:
*   `--source`: 音频文件路径列表。
*   `--target`: 参考文本路径列表。
*   `--agent-script`: 包含 Agent 类的 Python 文件路径。
*   `--agent-class`: Agent 类名。
*   `--task`: `s2t` 或 `s2s`。
*   `--computation-aware`: 开启计算感知延迟（统计模型推理耗时）。
*   `--quality`: **(推荐)** 跑完延迟后，自动调用 `TranslationEvaluator` 计算质量指标。
*   `--visualize`: 生成延迟阶梯图 (需安装 `[viz]`)。

### 3. S2S 进阶：MFA 强制对齐

对于 **Speech-to-Speech** 任务，为了精确计算延迟（基于生成的音频内容而非切片），本工具支持调用 **Montreal Forced Aligner (MFA)**。

**前置要求**:
1.  安装 MFA: `conda install -c conda-forge montreal-forced-aligner`
2.  下载模型: `mfa model download dictionary english_mfa` & `mfa model download acoustic english_mfa`

**使用方法**:
只需在运行 S2S 任务时，脚本会自动检测 MFA 环境。如果环境就绪，会自动计算额外的对齐指标：
*   `StartOffset_SpeechAlign`
*   `ATD_SpeechAlign`
---

## 🎭 模块三：多模态情感综合评测 (`EmotionEvaluator`)

专为 **同传/翻译情感保留度** 和 **情感生成准确率** 任务设计。基于前沿开源声音情感大模型 `iic/emotion2vec_plus_large`，提供纯音频驱动的两大维度支持：

1. **跨语种情感保真度评测 (Emotion Fidelity)**：基于大模型提取的 768-d 高维情感特征，计算原音频和生成音频在情感空间中的特征余弦相似度 (Cosine Similarity)。
2. **离散情感分类评测 (Classification Accuracy)**：将大模型用作 Zero-shot 分类器，计算生成声音的情感分类与参考目标标签的准确率（Accuracy）。

### 1. 初始化

默认会自动从 ModelScope 加载基准模型。支持离线加载与自定义映射：

```python
from multimetriceval import EmotionEvaluator

# 初始化评测器
evaluator = EmotionEvaluator(
    model_id="iic/emotion2vec_plus_large",    # 默认使用 Emotion2Vec+ large 模型
    device="cuda",                            # 自动探测，或手动指定 "cuda" / "cpu"
    custom_label_map={"hap": "happy", "exc": "excited"} # 将模型识别标签或自定义标签统一强映射
)
```

### 2. 动态调度：数据准备与评测

工具依据你传入的方法参数自动决定执行哪种评测（或同时执行两种）。全过程**纯音频**评测，不再依赖对应文本。

#### 场景 A：情感保真度 (Fidelity)
只传 `target_audio` 和 `source_audio` 即可触发保真度测算：

```python
results = evaluator.evaluate_all(
    source_audio="./data/src_wavs/",  # 原音频（如：源语言音频）
    target_audio="./data/tgt_wavs/",  # 目标音频（如：翻译/变声后的音频）
    verbose=True
)

# 返回结果包含：
# - Emotion_Fidelity_Cosine (综合情感特征余弦相似度，越接近 1 则情感越保真)
```

#### 场景 B：离散情感识别准确率 (Classification)
只传目标音频 `target_audio` 与基准标签 `reference_labels` 即可触发分类测算：

```python
results = evaluator.evaluate_all(
    target_audio="./data/tgt_wavs/",                 # 生成的音频
    reference_labels=["happy", "sad", "neutral"]     # 参考的情感标签列表，也可传 .json 路径
)

# 返回结果包含：
# - Emotion_Accuracy (模型识别得到的离散情感准确度)
```
> **💡 智能推演：** 假如你只传入了 `source_audio` 和 `reference_labels`，引擎会自动将其回退判定为你正在评测原音频本身的分类准确度。

### 3. JSON 数据格式支持
如果你的数据存储在 JSON 中，工具也能智能提取相关字段：

```bash
# target_data_config.json 示例:
# [{"audio": "path/to/1.wav", "label": "happy"}, ...]
```

```python
results = evaluator.evaluate_all(
    source_audio=["src1.wav", "src2.wav"],
    target_audio="target_data_config.json", # 工具会自动寻找 "audio" 的音频路径
    reference_labels="target_data_config.json" # 工具会自动寻找 "label" 进行校验
)
```
## 🗣️ 模块四：副语言与声学事件评测 (`ParalinguisticEvaluator`)

专为 **同传/S2ST翻译中非语言事件（Paralinguistics）的保留度** 设计。
默认基于前沿的开源多模态环境声音分析大模型 `laion/clap-htsat-fused`，提供双轨制评测逻辑：

1. **宏观连续特征保真度 (Continuous Fidelity Cosine)**：基于 CLAP 提取源与目标音频的 512 维最深层空间泛环境/声音特征的向量余弦相似度。
2. **微观离散物理事件保留率 (Discrete Retention F1)**：基于预设描述文本（如 "laughter", "coughing"），让 CLAP 在 Zero-Shot 环境下捕捉特定事件的有无，计算目标集合相对于源集合的重叠情况 (F1 数值)。

### 1. 初始化

默认从 HuggingFace 加载 `laion/clap-htsat-fused`，支持开启或关闭特定轨道的测算：

```python
from multimetriceval.paralinguistic_evaluator import ParalinguisticEvaluator

evaluator = ParalinguisticEvaluator(
    use_continuous_fidelity=True,  # 是否计算宏观声学特征余弦保真度
    use_discrete_matching=True,    # 是否计算微观离散声学事件捕获匹配度(F1)
    clap_model_path=None,          # 可自定义为已下载的本地离线模型路径
    device="cuda"                  # 自动探测或手动指定 "cuda" / "cpu"
)
```

### 2. 丰富的数据载入方式 (Data Formats)

`evaluate_all()` 方法具有极高容错性，支持通过以下**四种不同方式**传入 `source_audio` (原音频) 和 `target_audio` (翻译生成的音频) 列表。唯一要求是：**双端对齐，数目保持一致**。

*   **格式 A: 纯文件夹路径输入（推荐）**
    只需传入两个文件夹的路径，工具会自动扫描其中的 `.wav`/`.mp3`/`.flac` 文件，并**按名字字母顺序对其排序配对**。
    ```python
    results = evaluator.evaluate_all(
        source_audio="./data/S2ST_source_audios/", 
        target_audio="./outputs/S2ST_target_audios/"
    )
    ```

*   **格式 B: JSON 格式配置文件**
    如果你有一个 JSON 列表（或字典下含列表），系统会自动遍历它，提取其中名为 `audio`, `path`, `file` 等通用键值。
    ```python
    # data.json -> [{"id": 0, "audio": "src/1.wav"}, {"id": 1, "audio": "src/2.wav"}]
    results = evaluator.evaluate_all(
        source_audio="source_data.json", 
        target_audio="target_data.json"
    )
    ```

*   **格式 C: 预初始化的 Python 列表**
    直接在脚本中定义好完整的音频绝对路径。
    ```python
    results = evaluator.evaluate_all(
        source_audio=["./src/a.wav", "./src/b.wav"], 
        target_audio=["./tgt/a.wav", "./tgt/b.wav"]
    )
    ```

*   **格式 D: 换行符隔开的 `.txt` 清单文件**
    直接读取保存有路径列表的文本文件。

### 3. 微调事件探测列表 (可选项)

系统默认内置的探测事件为：`["laughter", "coughing", "sighing", "breathing heavily", "throat clearing"]`。
如果你在处理特定的 Benchmark（如 WESR 带特定的哭泣声 `crying` 标签），你可以在库源码 `_pseudo_detect_events` 逻辑中修改 `candidate_events` 变量进行完全自适应。


---

## 🛡️ 离线 / 内网环境使用指南 (语音模型)

如果在无法连接 GitHub 或 HuggingFace 的服务器（如校园网 HPC）上使用，请按以下步骤操作。

### 1. 准备 UTMOS 模型
UTMOS 依赖 GitHub 源码加载。
1.  **源码**: 下载并解压 [SpeechMOS 源码](https://github.com/tarepan/SpeechMOS/archive/refs/heads/main.zip) 到本地（例如 `/path/to/SpeechMOS-main`）。
2.  **权重**: 下载 [utmos22_strong.pth](https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/utmos22_strong.pth) 到本地任意位置（例如 `/path/to/utmos22_strong.pth`）。

### 2. 准备 Whisper 模型
1.  下载权重文件（如 [medium.pt](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc6017195dc-medium.pt)）。
2.  放入文件夹（例如 `/path/to/whisper_weights/`）。

### 3. 代码调用

通过参数显式指定路径，**无需**将文件放入系统缓存目录。

```python
evaluator = TranslationEvaluator(
    use_utmos=True,
    use_whisper=True,
    # [新增] 指定本地源码路径
    utmos_model_path="/path/to/SpeechMOS-main",
    # [新增] 指定本地权重路径 (.pth)
    utmos_ckpt_path="/path/to/utmos22_strong.pth", 
    # 指定 Whisper 权重文件路径 (注意这里要带上 .pt 文件名或确保 load_model 能找到)
    whisper_model="medium", 
    # 或者通过环境变量控制 whisper 缓存路径
)
```
*(注：Whisper 的 `load_model` 的 `download_root` 参数在 `TranslationEvaluator` 内部目前使用的是默认缓存或 `~/.cache/whisper`，如需完全离线指定路径，建议提前设置环境变量或将模型放入默认缓存目录)*

### 4. 准备 Emotion (情感) 模型

EmotionEvaluator 基于开源大模型 `emotion2vec_plus_large`，主要依赖 `funasr` 和 `modelscope` 进行加载。如果在断网环境下使用，可以通过本地路径加载：

1. 从 ModelScope (魔搭社区) 下载 `iic/emotion2vec_plus_large` 模型的完整仓库到本地文件夹。
2. 在初始化时，将 `model_id` 替换为存放模型的绝对路径。

```python
evaluator = EmotionEvaluator(
    # 指定本地离线模型文件夹路径
    model_id="/path/to/local/emotion2vec_plus_large", 
    device="cuda"
)
```

### 5. 准备 Paralinguistic (副语言/CLAP) 模型
由于 HPC 集群网络限制往往无法直连 HuggingFace：
1. **本地下载**: 在有代理的机器上运行 Python 脚本：
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id="laion/clap-htsat-fused", local_dir="./local_clap_model")
   ```
2. **离线加载**: 将模型上传至集群，在评测时指定物理路径：
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

| 模块 | 指标 | 全称 | 用途 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **TranslationEvaluator** | **sacreBLEU** | Bilingual Evaluation Understudy | 翻译 n-gram 匹配度 | 传统指标 |
| | **chrF++** | Character n-gram F-score | 字符级匹配度 | 适合形态丰富的语言 |
| | **COMET** | Crosslingual Optimized Metric | 基于神经网络的语义相似度 | **推荐** |
| | **BLEURT** | Bilingual Evaluation Understudy with Representations | 基于 BERT 的语义评分 | Google 出品 |
| | **UTMOS** | UTMOS22 Strong | 语音自然度/MOS 预测 | 无需参考音频 |
| | **WER/CER** | Word/Character Error Rate | 识别准确率 | **自动适配中英文** |
| **LatencyEvaluator**| **StartOffset** | Start Offset | 首字/首音延迟 | 反应速度 |
| | **ATD** | Average Token Delay | 平均 Token 延迟 | 综合延迟指标 |
| **EmotionEvaluator**| **Emotion Fidelity**| Emotion Fidelity (Cosine) | 跨语种声音情感保真度 | 基于 Emotion2Vec+ 高维特征 |
| | **Emotion Accuracy**| Emotion Classification Accuracy | 离散情感分类准确率 | 基于大模型提取特征的泛化识别 |
| **ParalinguisticEvaluator**| **Paralinguistic Fidelity**| Paralinguistic Fidelity (Cosine)| 连续环境声学质感保真度 | 基于多模态 CLAP 判断整体氛围对齐度 |
| | **Event Retention F1**| Event Retention Rate (F1) | 离散非语言物理发声存活率 | 测试笑声、叹气、咳嗽等是否被保留 |

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