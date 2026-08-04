#! /usr/bin/env python
import argparse
import os
import time
import socket

# huggingface_hub 会在导入时读取部分环境变量,因此必须先配置镜像与超时.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)

"""
文件说明:
    利用 huggingface_hub.snapshot_download 方法, 预先下载模型权重到本地缓存.
    强制忽略不安全的 .bin 权重,只下载 .safetensors
"""

# 强制设置全局 Socket 超时时间 (例如 60 秒).
# 如果 60 秒内没有任何数据传输,底层就会抛出 socket.timeout 异常,
# 从而成功触发下面的 except 逻辑.
socket.setdefaulttimeout(60)

# 想要预先下载的模型列表
MODELS = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "NousResearch/Meta-Llama-3.1-8B",
    "mistralai/Ministral-8B-Instruct-2410",
]


def robust_download(model_name, max_retries=100):
    """
    专门负责下载的函数
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
            if retries >= max_retries:
                print(
                    f"[Error] {model_name} 达到最大重试次数 ({max_retries}),跳过该模型."
                )
                break
            print(f"正在尝试第 {retries} 次重连 (5秒后)...")
            time.sleep(5)


def touch_model(model_name):
    """
    负责加载和校验
    """
    print(f"[Loading] {model_name}")
    try:
        config = AutoConfig.from_pretrained(
            model_name, local_files_only=True, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True, trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # 替换为主流的 BF16 数据格式
            local_files_only=True,
            trust_remote_code=True,
            config=config,
            device_map="auto",
            use_safetensors=True,
        )
        print(f"[OK] 成功加载并校验: {model_name} (Safetensors 格式, BF16)")

        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[Error] 加载失败 {model_name}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="预下载并可选校验 Hugging Face 模型")
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="只处理指定模型;可重复传入.未指定时处理 MODELS 中的全部模型",
    )
    parser.add_argument(
        "--skip-load-check",
        action="store_true",
        help="仅下载到缓存,不将完整模型加载到设备进行校验",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    models = args.models or MODELS
    print(f"Current HF_HOME: {os.getenv('HF_HOME')}")
    print(f"HF_ENDPOINT: {os.getenv('HF_ENDPOINT')}")
    for m in models:
        robust_download(m)
        if not args.skip_load_check:
            touch_model(m)
