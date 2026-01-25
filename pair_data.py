import json
import os

# 配置路径
INPUT_JSON = "./internal_data/data.json"
OUTPUT_JSON = "./internal_data/dataset_paired.json"

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"❌ 找不到 {INPUT_JSON}")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    total_count = len(raw_data)
    
    # 基础校验：总数必须是偶数才能两两配对
    if total_count % 2 != 0:
        print(f"⚠️ 警告: 数据总数是 {total_count} (奇数)，无法完美拆分为两半进行配对！")
        print("   -> 请检查原始数据，确保中英文文件数量一致。")
        # 如果你依然想强行运行，可以去掉下面的 return，但这会导致最后一个文件落单
        return

    half_count = total_count // 2
    paired_data = []

    print(f"📊 数据总数: {total_count}")
    print(f"   - 前半部分 (英文参考): 0 ~ {half_count-1}")
    print(f"   - 后半部分 (中文源):   {half_count} ~ {total_count-1}")
    print("-" * 40)

    for i in range(half_count):
        # 英文文件索引 (前半段)
        idx_eng = i
        # 中文文件索引 (后半段)
        idx_zh = i + half_count
        
        english_entry = raw_data[idx_eng]
        chinese_entry = raw_data[idx_zh]

        print(f"🔗 配对 [{i+1}/{half_count}]: "
              f"Source({chinese_entry['audio_file']}) <==> Ref({english_entry['audio_file']})")

        # 构建评测用的数据项
        # 目标: 中文(Source) -> 翻译成 -> 英文(Reference)
        pair = {
            "id": f"pair_{i + 1:04d}",
            
            # Source 部分 (题目)：给用户的中文语音和文本
            "source_speech_path": f"./internal_data/audio/{chinese_entry['audio_file']}", 
            
            # 这里务必确保 chinese_entry['reference_text'] 已经被人工修改为正确的中文文本
            "source_text": chinese_entry['reference_text'], 
            
            # Reference 部分 (答案)：用于评测的标准英文文本
            # 这里务必确保 english_entry['reference_text'] 已经被人工修改为正确的英文文本
            "reference_text": english_entry['reference_text'] 
        }
        
        paired_data.append(pair)

    # 保存
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(paired_data, f, ensure_ascii=False, indent=4)
        
    print("-" * 40)
    print(f"✅ 配对完成！共生成 {len(paired_data)} 组测试数据。")
    print(f"📂 新文件保存在: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()