from statistics import mean
from typing import Dict, List, Union
from .instance import SpeechToTextInstance, SpeechToSpeechInstance
from .utils import Aligner

SCORERS = {}
def register(name):
    def _reg(cls):
        SCORERS[name] = cls
        return cls
    return _reg

def with_alignment(scorer_cls):
    class AlignedScorer(scorer_cls):
        def __init__(self, computation_aware=False, output_dir="./output"):
            super().__init__(computation_aware)
            self.aligner = Aligner(output_dir)

        def __call__(self, instances):
            self.aligner.run_mfa()
            for idx, ins in instances.items():
                if not isinstance(ins, SpeechToSpeechInstance): continue
                start_offset = ins.delays[0] if ins.delays else 0
                aligned = self.aligner.get_alignment_delays(idx, start_offset)
                if aligned:
                    ins.raw_delays = ins.delays 
                    ins.delays = aligned
            return super().__call__(instances)
    return AlignedScorer

class LatencyScorer:
    def __init__(self, computation_aware: bool = False):
        self.computation_aware = computation_aware
    def subtract(self, arr1, arr2):
        return [x - y for x, y in zip(arr1, arr2)]
    def get_delays(self, ins):
        return ins.elapsed if self.computation_aware else ins.delays

@register("StartOffset")
class StartOffset(LatencyScorer):
    def __call__(self, instances):
        scores = []
        for ins in instances.values():
            d = self.get_delays(ins)
            scores.append(d[0] if d else 0)
        return mean(scores) if scores else 0

@register("StartOffset_SpeechAlign")
@with_alignment
class StartOffsetAligned(StartOffset):
    pass

@register("ATD")
class ATDScorer(LatencyScorer):
    def __call__(self, instances) -> float:
        scores = []
        for index, ins in instances.items():
            SRC_TOKEN_LEN = 300
            if isinstance(ins, SpeechToSpeechInstance):
                TGT_TOKEN_LEN = 300
                OUTPUT_TYPE = "speech"
            else:
                TGT_TOKEN_LEN = 0 
                OUTPUT_TYPE = "text"

            delays = getattr(ins, "delays", None)
            if not delays: continue

            if self.computation_aware:
                elapsed = getattr(ins, "elapsed", None)
                compute_elapsed = self.subtract(elapsed, ins.delays)
                compute_times = self.subtract(compute_elapsed, [0] + compute_elapsed[:-1])
            else:
                compute_times = [0] * len(delays)

            chunk_sizes = {"src": [0], "tgt": [0]}
            token_to_chunk = {"src": [0], "tgt": [0]}
            token_to_time = {"src": [0], "tgt": [0]}
            
            tgt_token_lens = []
            delays_no_duplicate = sorted(list(set(delays)))

            if OUTPUT_TYPE == "text":
                for i in range(len(delays)):
                    chunk_sizes["tgt"].append(1)
                    token_to_chunk["tgt"].append(i+1)
                tgt_token_lens = [TGT_TOKEN_LEN] * len(delays)
            else:
                s2s_delays, s2s_comps = [], []
                for i, dur in enumerate(ins.durations):
                    num, rest = divmod(dur, TGT_TOKEN_LEN)
                    tokens = int(num)*[TGT_TOKEN_LEN] + ([rest] if rest>0 else [])
                    tgt_token_lens += tokens
                    chunk_sizes["tgt"].append(len(tokens))
                    token_to_chunk["tgt"] += [i+1] * len(tokens)
                    s2s_delays += [delays[i]] * len(tokens)
                    s2s_comps += [compute_times[i]/len(tokens)] * len(tokens)
                delays = s2s_delays
                compute_times = s2s_comps

            src_durations = self.subtract(delays_no_duplicate, [0] + delays_no_duplicate[:-1])
            for i, dur in enumerate(src_durations, 1):
                num, rest = divmod(dur, SRC_TOKEN_LEN)
                tokens = int(num)*[SRC_TOKEN_LEN] + ([rest] if rest>0 else [])
                chunk_sizes["src"].append(len(tokens))
                for t_len in tokens:
                    prev = token_to_time["src"][-1] if token_to_time["src"] else 0
                    token_to_time["src"].append(prev + t_len)
                    token_to_chunk["src"].append(i)

            for delay, comp, t_len in zip(delays, compute_times, tgt_token_lens):
                prev_tgt = token_to_time["tgt"][-1] if token_to_time["tgt"] else 0
                start = max(delay, prev_tgt)
                token_to_time["tgt"].append(start + t_len + comp)

            scores.append(self.compute_algo(chunk_sizes, token_to_chunk, token_to_time))
        return mean(scores) if scores else 0

    def compute_algo(self, chunk_sizes, token_to_chunk, token_to_time):
        tgt_to_src = []
        for t in range(1, len(token_to_chunk["tgt"])):
            chunk_id = token_to_chunk["tgt"][t]
            acc_x = sum(chunk_sizes["src"][:chunk_id])
            acc_y = sum(chunk_sizes["tgt"][:chunk_id])
            S = t - max(0, acc_y - acc_x)
            curr_src = sum(chunk_sizes["src"][:chunk_id+1])
            s_idx = S if S < curr_src else curr_src
            if s_idx < len(token_to_time["src"]):
                src_time = token_to_time["src"][s_idx]
            else:
                src_time = token_to_time["src"][-1]
            tgt_to_src.append((t, s_idx))

        atd_delays = []
        for t, s in tgt_to_src:
            s_safe = min(s, len(token_to_time["src"]) - 1)
            val = token_to_time["tgt"][t] - token_to_time["src"][s_safe]
            atd_delays.append(val)
        return float(mean(atd_delays)) if atd_delays else 0

@register("ATD_SpeechAlign")
@with_alignment
class ATDScorerAligned(ATDScorer):
    pass

@register("CustomATD")
class CustomATD(ATDScorer):
    """
    自定义 ATD 逻辑：在计算 Token 结束时间时，不累加 Token 自身的虚拟长度 (t_len)。
    这关注的是生成动作结束的时刻。
    """
    def __call__(self, instances) -> float:
        scores = []
        # 我们必须重写整个 __call__，因为差异在循环内部
        for index, ins in instances.items():
            SRC_TOKEN_LEN = 300
            if isinstance(ins, SpeechToSpeechInstance):
                TGT_TOKEN_LEN = 300
                OUTPUT_TYPE = "speech"
            else:
                TGT_TOKEN_LEN = 0 
                OUTPUT_TYPE = "text"

            delays = getattr(ins, "delays", None)
            if not delays: continue

            if self.computation_aware:
                elapsed = getattr(ins, "elapsed", None)
                compute_elapsed = self.subtract(elapsed, ins.delays)
                compute_times = self.subtract(compute_elapsed, [0] + compute_elapsed[:-1])
            else:
                compute_times = [0] * len(delays)

            chunk_sizes = {"src": [0], "tgt": [0]}
            token_to_chunk = {"src": [0], "tgt": [0]}
            token_to_time = {"src": [0], "tgt": [0]}
            
            tgt_token_lens = []
            delays_no_duplicate = sorted(list(set(delays)))

            if OUTPUT_TYPE == "text":
                for i in range(len(delays)):
                    chunk_sizes["tgt"].append(1)
                    token_to_chunk["tgt"].append(i+1)
                tgt_token_lens = [TGT_TOKEN_LEN] * len(delays)
            else:
                s2s_delays, s2s_comps = [], []
                for i, dur in enumerate(ins.durations):
                    num, rest = divmod(dur, TGT_TOKEN_LEN)
                    tokens = int(num)*[TGT_TOKEN_LEN] + ([rest] if rest>0 else [])
                    tgt_token_lens += tokens
                    chunk_sizes["tgt"].append(len(tokens))
                    token_to_chunk["tgt"] += [i+1] * len(tokens)
                    s2s_delays += [delays[i]] * len(tokens)
                    s2s_comps += [compute_times[i]/len(tokens)] * len(tokens)
                delays = s2s_delays
                compute_times = s2s_comps

            src_durations = self.subtract(delays_no_duplicate, [0] + delays_no_duplicate[:-1])
            for i, dur in enumerate(src_durations, 1):
                num, rest = divmod(dur, SRC_TOKEN_LEN)
                tokens = int(num)*[SRC_TOKEN_LEN] + ([rest] if rest>0 else [])
                chunk_sizes["src"].append(len(tokens))
                for t_len in tokens:
                    prev = token_to_time["src"][-1] if token_to_time["src"] else 0
                    token_to_time["src"].append(prev + t_len)
                    token_to_chunk["src"].append(i)

            # ============ 关键修改区域 ============
            for delay, comp, t_len in zip(delays, compute_times, tgt_token_lens):
                prev_tgt = token_to_time["tgt"][-1] if token_to_time["tgt"] else 0
                start = max(delay, prev_tgt)
                
                # 原版: start + t_len + comp
                # 修改版: start + comp (去掉了 t_len)
                token_to_time["tgt"].append(start + comp)
            # ====================================

            scores.append(self.compute_algo(chunk_sizes, token_to_chunk, token_to_time))
        return mean(scores) if scores else 0

@register("CustomATD_SpeechAlign")
@with_alignment
class CustomATDAligned(CustomATD):
    pass