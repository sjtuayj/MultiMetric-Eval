import os
import shutil
import json
import csv
import whisper
import warnings
import torch
import opencc  # <--- 1. 新增库导入

# ================= 🔴 必须修改的配置 🔴 =================
WHISPER_MODEL_PATH = "./model/whisper_medium/whisper_models/medium.pt" 
RAW_AUDIO_DIR = "./raw_audios"
# ========================================================

OUTPUT_DIR = "./internal_data"
OUTPUT_AUDIO_DIR = "./internal_data/audio"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"🚀 启动离线数据处理脚本 (设备: {DEVICE})")

    if not os.path.exists(WHISPER_MODEL_PATH):
        print(f"❌ 错误: 找不到 Whisper 模型文件: {WHISPER_MODEL_PATH}")
        return
    
    if not os.path.exists(RAW_AUDIO_DIR):
        print(f"❌ 错误: 找不到原始音频目录: {RAW_AUDIO_DIR}")
        return

    if not os.path.exists(OUTPUT_AUDIO_DIR):
        os.makedirs(OUTPUT_AUDIO_DIR)

    # 2. 离线加载 Whisper
    print(f"⏳ 正在从本地加载 Whisper 模型: {WHISPER_MODEL_PATH} ...")
    try:
        model = whisper.load_model(WHISPER_MODEL_PATH, device=DEVICE)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    print("✅ 模型加载成功！")

    # --- 🔵 新增：初始化繁简转换器 (t2s = Traditional to Simplified) ---
    # 这个转换器非常智能，它只动汉字，不动英文
    converter = opencc.OpenCC('t2s') 

    valid_exts = ('.wav', '.mp3', '.m4a', '.flac', '.ogg')
    files = sorted([f for f in os.listdir(RAW_AUDIO_DIR) if f.lower().endswith(valid_exts)])
    
    if not files:
        print(f"⚠️ 警告: 未找到音频文件。")
        return
        
    print(f"🔎 发现 {len(files)} 个音频文件，开始处理...")

    json_data = []
    rename_log = []

    for idx, old_filename in enumerate(files):
        ext = os.path.splitext(old_filename)[1]
        new_filename = f"audio_{idx+1:04d}{ext}"
        
        old_path = os.path.join(RAW_AUDIO_DIR, old_filename)
        new_path = os.path.join(OUTPUT_AUDIO_DIR, new_filename)

        shutil.copy2(old_path, new_path)
        rename_log.append([old_filename, new_filename])

        print(f"   [{idx+1}/{len(files)}] 转写中: {new_filename} ...", end="\r")
        
        try:
            # --- 🔵 关键修改：转写参数 ---
            result = model.transcribe(
                new_path, 
                fp16=(DEVICE=="cuda"),
                # ❌ 不要指定 language="zh"，否则英文会被强行翻译成中文
                # ❌ 不要加 "以下是简体中文" 这种强提示词
                # ✅ 可以加一个中英混合的提示词，告诉模型这里有混合语种，有助于断句
                initial_prompt="Hello, 你好。" 
            )
            raw_text = result["text"].strip()
            
            # --- 🔵 关键修改：强制转简体 ---
            # 这一步会将 raw_text 里的繁体字变成简体，同时保留英文原样
            text = converter.convert(raw_text)

        except Exception as e:
            print(f"\n   ⚠️ 转写失败 ({new_filename}): {e}")
            text = ""

        entry = {
            "id": f"{idx+1:04d}",
            "audio_file": new_filename,
            "source_text": text,
            "reference_text": text,
            "original_filename": old_filename
        }
        json_data.append(entry)

    print(f"\n✨ 所有音频处理完毕！")

    json_output_path = os.path.join(OUTPUT_DIR, "data.json")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    
    csv_output_path = os.path.join(OUTPUT_DIR, "filename_mapping.csv")
    with open(csv_output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["原始文件名", "新文件名"])
        writer.writerows(rename_log)

    print("="*40)
    print(f"✅ 完成！结果保存在: {OUTPUT_DIR}")
    print("="*40)

if __name__ == "__main__":
    main()