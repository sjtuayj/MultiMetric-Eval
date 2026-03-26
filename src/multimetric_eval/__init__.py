from .translation_evaluator import (
    TranslationEvaluator,
    load_text_from_file_or_list, # <--- 替换原来的 load_target_text_from_file
    load_audio_from_folder,
)

# 🔥 新增：导入刚才写好的 EmotionEvaluator
from .emotion_evaluator import EmotionEvaluator

from .dataset import (
    Dataset,
    load_dataset,
    list_datasets,
    get_dataset_info,
    create_dataset_from_json,
)

# 注意：SpeechEvaluator 已被移除，功能已合并入 TranslationEvaluator
# from .speech_evaluator import SpeechEvaluator 

from .latency.cli import LatencyEvaluator
from .latency.agent import GenericAgent, AgentPipeline
from .latency.basics import ReadAction, WriteAction

# 建议升级版本号，因为新增了重要的情感评测功能
__version__ = "0.6.0"  

__all__ = [
    # 核心评测器
    "TranslationEvaluator",      # 集成：翻译(BLEU/COMET) + 语音(UTMOS/WER)
    "EmotionEvaluator",          # 🔥 新增：情感评测 (SER)
    "LatencyEvaluator",
    "GenericAgent",
    "AgentPipeline",
    "ReadAction", 
    "WriteAction",
    
    # 工具函数
    "load_text_from_file_or_list", # <--- 更新
    "load_audio_from_folder",

    # 数据集管理
    "Dataset",
    "load_dataset",
    "list_datasets",
    "get_dataset_info",
    "create_dataset_from_json",
]