# environment
torch openai-whisper sacrebleu unbabel-comet bleurt

bleurt的Python 调用

from bleurt import score

scorer = score.BleurtScorer("BLEURT-20") # 填BLEURT-20的绝对路径

### 版本冲突

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.unbabel-comet 2.2.7 requires protobuf<5.0.0,>=4.24.4, but you have protobuf 6.33.4 which is incompatible.

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.tensorflow 2.20.0 requires protobuf>=5.28.0, but you have protobuf 4.25.8 which is incompatible.

当前是protobuf 4.25.8

# prepare_data.py

在./raw_audios已有.wav文件的情况下将源音频文件通过asr获得文本以及重新修改名称为0001、0002...并保存在./internal_data中。

# pair_data.py

在已有./internal_data/data.json的情况下，将中英文进行配对，为用于evaluator.py做准备，生成./internal_data/dataset_paired.json

