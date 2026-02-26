from .evaluator import (
    ModelEvaluator,
    load_hypothesis_from_file,
    load_audio_from_folder,
)
from .dataset import (
    Dataset,
    load_dataset,
    list_datasets,
    get_dataset_info,
    create_dataset_from_json,
)

from .speech_evaluator import SpeechEvaluator
from .latency.cli import LatencyEvaluator
from .latency.agent import GenericAgent, AgentPipeline
from .latency.basics import ReadAction, WriteAction

__version__ = "0.4.1"

__all__ = [
    # 核心评测器
    "ModelEvaluator",      # 翻译/语义评测 (BLEU, COMET, etc.)
    "SpeechEvaluator",     # 语音质量评测 (UTMOS, WER)
    "LatencyEvaluator",
    "GenericAgent",
    "AgentPipeline",
    "ReadAction", 
    "WriteAction",
    # 工具函数
    "load_hypothesis_from_file",
    "load_audio_from_folder",

    # 数据集管理
    "Dataset",
    "load_dataset",
    "list_datasets",
    "get_dataset_info",
    "create_dataset_from_json",
]