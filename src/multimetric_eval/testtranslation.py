import os
import json
import wave
import shutil
import unittest
import torch
from pathlib import Path

# 导入你的模块 (假设文件名是 translation_evaluator.py)
try:
    from translation_evaluator import TranslationEvaluator
except ImportError:
    print("❌ 错误: 找不到 translation_evaluator.py，请确保文件在当前目录下。")
    exit(1)

# ================= 配置 =================
# 为了测试速度，默认关闭重型模型 (COMET/BLEURT)，仅测试逻辑流程
# 如果你已经配置好环境想测模型，请改为 True
USE_COMET = False   
USE_BLEURT = False
USE_WHISPER = True  # 建议开启以测试音频流程
TEST_DIR = "./test_artifacts"

class TestTranslationEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """测试开始前：创建临时文件和目录"""
        print("\n🚀 === 开始初始化测试环境 ===")
        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)
        os.makedirs(TEST_DIR)

        # 1. 创建虚拟音频文件 (1秒静音 wav)
        cls.audio_dir = os.path.join(TEST_DIR, "audio")
        os.makedirs(cls.audio_dir)
        cls.wav_path = os.path.join(cls.audio_dir, "test_audio.wav")
        with wave.open(cls.wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b'\0' * 16000) # 写入1秒静音数据

        # 2. 创建翻译结果文件 (New JSON 格式 - target_text)
        cls.json_new_path = os.path.join(TEST_DIR, "output_new.json")
        with open(cls.json_new_path, 'w', encoding='utf-8') as f:
            json.dump({"target_text": ["The cat is on the mat."]}, f)

        # 3. 创建翻译结果文件 (Old JSON 格式 - hypothesis 兼容性测试)
        cls.json_old_path = os.path.join(TEST_DIR, "output_old.json")
        with open(cls.json_old_path, 'w', encoding='utf-8') as f:
            json.dump({"hypothesis": ["The cat is on the mat."]}, f)

        # 4. 创建 TXT 文件
        cls.txt_path = os.path.join(TEST_DIR, "output.txt")
        with open(cls.txt_path, 'w', encoding='utf-8') as f:
            f.write("The cat is on the mat.\n")

        # 初始化评测器
        print("⏳ 正在加载模型 (这可能需要一点时间)...")
        cls.evaluator = TranslationEvaluator(
            use_comet=USE_COMET,
            use_bleurt=USE_BLEURT,
            use_whisper=USE_WHISPER,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("✅ 模型加载完成")

    @classmethod
    def tearDownClass(cls):
        """测试结束后：清理垃圾文件"""
        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)
        print("\n🧹 测试环境清理完毕")

    def test_01_text_list_input(self):
        """测试 1: 基础列表输入 (使用新参数名 target_text)"""
        print("\n[测试 1] 纯文本列表输入...")
        results = self.evaluator.evaluate(
            target_text=["The cat is on the mat."],
            reference=["The cat is on the mat."],
            source=["猫在垫子上。"]
        )
        # 验证指标是否存在
        self.assertIn("sacreBLEU", results)
        self.assertIn("chrF++", results)
        # 完全匹配，BLEU 应该接近 100
        self.assertGreater(results["sacreBLEU"], 99.0)
        print("   ✅ 结果:", results)

    def test_02_file_input_json_new(self):
        """测试 2: 文件路径输入 (JSON target_text)"""
        print("\n[测试 2] 文件输入 (JSON target_text)...")
        results = self.evaluator.evaluate(
            target_text=self.json_new_path,  # 传入文件路径
            reference=["The cat is on the mat."],
            source=["猫在垫子上。"]
        )
        self.assertIn("target_text", results) # 验证是否返回了具体文本
        self.assertEqual(results["target_text"][0], "The cat is on the mat.")
        print("   ✅ 读取成功")

    def test_03_file_input_json_compatibility(self):
        """测试 3: JSON 兼容性 (读取旧字段 hypothesis)"""
        print("\n[测试 3] 旧格式 JSON 兼容性测试...")
        results = self.evaluator.evaluate(
            target_text=self.json_old_path, # 这是一个 key 为 hypothesis 的文件
            reference=["The cat is on the mat."],
            source=["猫在垫子上。"]
        )
        self.assertEqual(results["target_text"][0], "The cat is on the mat.")
        print("   ✅ 兼容性测试通过")

    def test_04_audio_input(self):
        """测试 4: 音频文件夹输入 (ASR模式)"""
        if not USE_WHISPER:
            print("\n[测试 4] 跳过 (未启用 Whisper)")
            return

        print("\n[测试 4] 音频输入测试 (ASR)...")
        results = self.evaluator.evaluate(
            target_speech=self.audio_dir, # 传入文件夹
            reference=["The cat is on the mat."],
            source=["猫在垫子上。"]
        )
        
        self.assertIn("sacreBLEU_ASR", results)
        self.assertIn("target_text_ASR", results)
        print(f"   ✅ ASR 识别结果: {results['target_text_ASR']}")
        # 注意：因为输入的是静音 wav，Whisper 可能会输出空字符串或幻觉，这里只测流程不报错即可

    def test_05_mixed_input(self):
        """测试 5: 双模式 (文本 + 音频)"""
        if not USE_WHISPER:
            return

        print("\n[测试 5] 混合模式输入...")
        results = self.evaluator.evaluate(
            target_text=["The cat is on the mat."],
            target_speech=self.audio_dir,
            reference=["The cat is on the mat."],
            source=["猫在垫子上。"]
        )
        # 应该同时包含两类指标
        self.assertIn("sacreBLEU", results)      # 文本指标
        self.assertIn("sacreBLEU_ASR", results)  # 语音指标
        print("   ✅ 混合模式通过")

    def test_06_dataset_helper(self):
        """测试 6: Dataset 辅助函数"""
        print("\n[测试 6] Dataset 辅助函数...")
        
        # 模拟一个 Dataset 对象
        class MockDataset:
            def __init__(self):
                self.reference_texts = ["The cat is on the mat."]
                self.source_texts = ["猫在垫子上。"]
        
        ds = MockDataset()
        results = self.evaluator.evaluate_dataset(
            dataset=ds,
            target_text=["The cat is on the mat."]
        )
        self.assertGreater(results["sacreBLEU"], 99.0)
        print("   ✅ Dataset 接口通过")

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)