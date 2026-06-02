#! /usr/bin/env python
import os
import time
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

"""
文件说明:
    利用 huggingface_hub.snapshot_download 方法, 预先下载模型权重到本地缓存.
    强制忽略不安全的 .bin 权重,只下载 .safetensors
"""

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # 关掉容易卡死的加速器
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 换成稳定的镜像源


# 想要预先下载的模型列表
MODELS = [
    "Qwen/Qwen3-4B",
]


def robust_download(model_name):
    """
    专门负责下载的函数:
    """
    print(f"\n[Download] 正在检查/下载: {model_name}")
    retries = 0
    while True:
        try:
            snapshot_download(
                repo_id=model_name,
                ignore_patterns=[
                    "*.flax",
                    "*.h5",
                    "*.tflite",
                    "*.msgpack",
                    "*.bin",
                    "*.pt",
                ],
            )
            print(f"[Download] ✅ {model_name} 准备就绪")
            break
        except Exception as e:
            retries += 1
            print(f"[Warning] 下载中断: {e}")
            print(f"正在尝试第 {retries} 次重连 (5秒后)...")
            time.sleep(5)


def touch_model(model_name):
    """
    负责加载和校验
    """
    print(f"[Loading] {model_name}")
    try:
        config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            local_files_only=True,
            config=config,
            device_map="auto",
            use_safetensors=True,
        )
        print(f"[OK] 成功加载并校验: {model_name} (Safetensors 格式)")

        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[Error] 加载失败 {model_name}: {e}")


if __name__ == "__main__":
    print(f"Current HF_HOME: {os.getenv('HF_HOME')}")
    for m in MODELS:
        robust_download(m)
        touch_model(m)
