#! /usr/bin/env python
import os
import time
import torch
import socket
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

"""
文件说明:
    利用 huggingface_hub.snapshot_download 方法, 预先下载模型权重到本地缓存.
    强制忽略不安全的 .bin 权重,只下载 .safetensors
"""

# 强制设置全局 Socket 超时时间 (例如 60 秒).
# 如果 60 秒内没有任何数据传输,底层就会抛出 socket.timeout 异常,
# 从而成功触发下面的 except 逻辑.
socket.setdefaulttimeout(60)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # 关掉容易卡死的加速器
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 换成稳定的镜像源
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"  # 增加 HF 专属下载超时设置

# 想要预先下载的模型列表
MODELS = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "NousResearch/Meta-Llama-3-8B",
    "mistralai/Ministral-3-8B-Instruct-2512",
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


def touch_model_new(model_name):
    """
    负责加载和校验:加入了智能 Fallback 机制和特定模型修复
    """
    print(f"\n[Loading] {model_name}")
    try:
        # 1. 动态处理 Tokenizer 参数:专门消除 Mistral 的正则警告
        tok_kwargs = {"local_files_only": True}
        if "istral" in model_name.lower():  # 命中 Mistral 或 Ministral
            tok_kwargs["fix_mistral_regex"] = True

        # 2. 优先尝试"原生加载" (trust_remote_code=False)
        # 因为我们更新了 transformers,主流模型原生支持更稳,避免远程代码冲突
        try:
            config = AutoConfig.from_pretrained(
                model_name, local_files_only=True, trust_remote_code=False
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=False, **tok_kwargs
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                trust_remote_code=False,
                config=config,
                device_map="auto",
                use_safetensors=True,
            )
            print(f"[OK] 成功加载并校验: {model_name} (原生支持, BF16)")

        # 3. 如果原生不支持,自动降级使用远程代码 (trust_remote_code=True)
        except Exception as native_e:
            print(f"  -> 原生加载失败,尝试启用 trust_remote_code... ({native_e})")

            config = AutoConfig.from_pretrained(
                model_name, local_files_only=True, trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, **tok_kwargs
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                trust_remote_code=True,
                config=config,
                device_map="auto",
                use_safetensors=True,
            )
            print(f"[OK] 成功加载并校验: {model_name} (Remote Code 模式, BF16)")

        # 验证完后立刻清理 A100 的显存,准备加载下一个
        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[Error] 加载彻底失败 {model_name}: {e}")


if __name__ == "__main__":
    print(f"Current HF_HOME: {os.getenv('HF_HOME')}")
    for m in MODELS:
        robust_download(m)
        touch_model_new(m)
