from .translation_evaluator import (
    TranslationEvaluator,
    load_text_from_file_or_list, 
    load_audio_from_folder,
)


from .emotion_evaluator import EmotionEvaluator
from .paralinguistic_evaluator import (
    DiscreteEventConfig,
    EventSpan,
    ParalinguisticEvaluator,
    ParalinguisticSample,
    build_paralinguistic_inputs,
    evaluate_paralinguistic_dataset,
    load_paralinguistic_samples,
    load_paralinguistic_manifest,
)
from .speech_quality_evaluator import SpeechQualityEvaluator
from .speaker_similarity_evaluator import SpeakerSimilarityEvaluator

from .dataset import (
    Dataset,
    load_dataset,
    list_datasets,
    get_dataset_info,
    create_dataset_from_json,
)


from .latency.cli import LatencyEvaluator
from .latency.agent import GenericAgent, AgentPipeline
from .latency.basics import ReadAction, WriteAction

__version__ = "0.8.2"  

__all__ = [
    # 核心评测器
    "TranslationEvaluator",     
    "EmotionEvaluator",         
    "ParalinguisticEvaluator",
    "DiscreteEventConfig",
    "EventSpan",
    "ParalinguisticSample",
    "load_paralinguistic_manifest",
    "load_paralinguistic_samples",
    "build_paralinguistic_inputs",
    "evaluate_paralinguistic_dataset",
    "SpeechQualityEvaluator",
    "LatencyEvaluator",
    "SpeakerSimilarityEvaluator",
    "GenericAgent",
    "AgentPipeline",
    "ReadAction", 
    "WriteAction",
    
    # 工具函数
    "load_text_from_file_or_list", 
    "load_audio_from_folder",

    # 数据集管理
    "Dataset",
    "load_dataset",
    "list_datasets",
    "get_dataset_info",
    "create_dataset_from_json",
]
