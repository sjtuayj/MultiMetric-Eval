import os
import shutil
import subprocess
import sys
import textgrid
import matplotlib.pyplot as plt
from pathlib import Path

def submit_slurm(args, python_script_path):
    # 简单的 Slurm 提交辅助函数
    cmd = f"python {python_script_path} " + " ".join([a for a in sys.argv[1:] if "--slurm" not in a])
    script = f"""#!/bin/bash
#SBATCH --job-name=latency_eval
#SBATCH --output={args.output}/slurm.log
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4

mkdir -p {args.output}
{cmd}
"""
    script_path = Path(args.output) / "run.sh"
    with open(script_path, "w") as f: f.write(script)
    print(f"Submitting Slurm job: {script_path}")
    os.system(f"sbatch {script_path}")
    sys.exit(0)

class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir) / "visual"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot(self, instance_data):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not installed, skipping visualization.")
            return

        data = instance_data
        idx = data["index"]
        delays = [d/1000 for d in data["delays"]]
        if not delays: return
        
        pred = data["prediction"]
        is_text = not str(pred).endswith(".wav")
        
        plt.figure(figsize=(10, 6))
        x_points = [0] + delays
        y_points = list(range(len(x_points)))
        plt.step(x_points, y_points, where='post', marker='o')
        plt.xlabel("Source Time (s)")
        plt.ylabel("Output Unit")
        plt.title(f"Instance {idx}")
        
        if is_text:
            words = pred.split()
            for i, txt in enumerate(words):
                if i+1 < len(delays):
                    plt.text(delays[i+1], i+1, txt)
        else:
            plt.text(0, 0, f"Saved to {pred}")

        plt.grid(True)
        plt.savefig(self.output_dir / f"{idx}.png")
        plt.close()

class Aligner:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.align_dir = self.output_dir / "align"
        self.wav_dir = self.output_dir / "wavs"
        self.temp_dir = self.output_dir / "mfa_temp"

    def run_mfa(self):
        if self.align_dir.exists() and any(self.align_dir.iterdir()):
            return
        try:
            subprocess.check_output("mfa version", shell=True)
        except:
            print("Error: 'mfa' command not found. Cannot run alignment.")
            return

        if self.temp_dir.exists(): shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir()
        self.align_dir.mkdir(parents=True, exist_ok=True)
        
        # 假设 english_mfa 模型已下载
        cmd = (
            f"mfa align {self.wav_dir} english_mfa english_mfa {self.align_dir} "
            f"--clean --overwrite --temporary_directory {self.temp_dir} --verbose"
        )
        try:
            subprocess.run(cmd, shell=True, check=True)
        except Exception as e:
            print(f"MFA Alignment failed: {e}")

    def get_alignment_delays(self, index, offset_delay):
        tg_path = self.align_dir / f"{index}_pred.TextGrid"
        if not tg_path.exists(): return None
        tg = textgrid.TextGrid.fromFile(tg_path)
        words_tier = tg[0] 
        aligned_delays = []
        for interval in words_tier:
            if interval.mark:
                aligned_delays.append(offset_delay + interval.maxTime * 1000)
        return aligned_delays