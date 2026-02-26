import time
import math
import soundfile
from pathlib import Path
from .basics import EmptySegment, SpeechSegment, Segment

class Instance:
    def __init__(self, index: int, source_path, reference=None, output_dir="./output"):
        self.index = index
        self.source_path = source_path
        self.reference = reference
        self.output_dir = Path(output_dir)
        self.reset()

    def reset(self):
        self.step = 0
        self.elapsed = [] 
        self.delays = [] 
        self.prediction_list = [] 
        self.start_time = None
        self.durations = [] # S2S 专用
        self.finish_prediction = False

    def summarize(self):
        return {
            "index": self.index,
            "source": (self.source_path, ""),
            "reference": self.reference,
            "prediction": self.get_prediction_content(),
            "prediction_length": len(self.prediction_list),
            "delays": self.delays,
            "elapsed": self.elapsed,
            "durations": self.durations
        }
    
    def get_prediction_content(self): raise NotImplementedError
    def get_prediction_raw(self): return self.prediction_list # 返回原始 list 供质量评测

class SpeechToTextInstance(Instance):
    def __init__(self, index, source_path, reference=None, output_dir="./output"):
        data, sr = soundfile.read(source_path, dtype="float32")
        self.samples = data.tolist()
        self.sample_rate = sr
        self.source_finished_reading = False
        super().__init__(index, source_path, reference, output_dir)

    def len_sample_to_ms(self, length): return length * 1000 / self.sample_rate

    def send_source(self, segment_size=10):
        if self.step == 0: self.start_time = time.time()
        num_samples = math.ceil(segment_size / 1000 * self.sample_rate)
        
        if self.step < len(self.samples):
            chunk = self.samples[self.step : self.step + num_samples]
            is_finished = (self.step + num_samples >= len(self.samples))
            self.step += len(chunk)
            return SpeechSegment(content=chunk, sample_rate=self.sample_rate, finished=is_finished)
        else:
            self.source_finished_reading = True
            return EmptySegment(finished=True)

    def receive_prediction(self, seg: Segment):
        if self.finish_prediction or seg.is_empty:
            self.finish_prediction = seg.finished
            return
        if not seg.content: return
        
        current_time = time.time()
        content_list = seg.content.strip().split()
        self.prediction_list += content_list
        
        curr_delay = self.len_sample_to_ms(self.step)
        curr_elapsed = curr_delay + (current_time - self.start_time) * 1000
        
        self.delays += [curr_delay] * len(content_list)
        self.elapsed += [curr_elapsed] * len(content_list)
        self.finish_prediction = seg.finished

    def get_prediction_content(self): return " ".join(self.prediction_list)
    
    @property
    def source_length(self): return self.len_sample_to_ms(len(self.samples))
    @property
    def reference_length(self): return len(self.reference.split()) if self.reference else 0

class SpeechToSpeechInstance(SpeechToTextInstance):
    def __init__(self, index, source_path, reference=None, output_dir="./output"):
        super().__init__(index, source_path, reference, output_dir)
        self.target_sample_rate = 16000

    def receive_prediction(self, seg: Segment):
        if self.finish_prediction or seg.is_empty:
            self.finish_prediction = seg.finished
            return
        if not seg.content: return
        if not self.start_time: self.start_time = time.time()
        
        current_time = time.time()
        duration_ms = len(seg.content) * 1000 / seg.sample_rate
        self.target_sample_rate = seg.sample_rate
        
        self.prediction_list.append(seg.content)
        self.durations.append(duration_ms)
        
        curr_delay = self.len_sample_to_ms(self.step)
        curr_elapsed = curr_delay + (current_time - self.start_time) * 1000
        
        self.delays.append(curr_delay)
        self.elapsed.append(curr_elapsed)
        self.finish_prediction = seg.finished

    def get_prediction_content(self):
        # 返回 wav 路径
        wav_dir = self.output_dir / "wavs"
        wav_dir.mkdir(parents=True, exist_ok=True)
        wav_path = wav_dir / f"{self.index}_pred.wav"
        
        flat_samples = [item for sublist in self.prediction_list for item in sublist]
        if flat_samples:
            soundfile.write(wav_path, flat_samples, self.target_sample_rate)
        return str(wav_path.absolute())
    
    def get_prediction_raw(self):
        # S2S 的原始预测结果是音频路径
        return self.get_prediction_content()

    @property 
    def reference_length(self): return len(self.prediction_list)