#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量下载并校验 Hugging Face 模型.

功能:
1. 使用 huggingface_hub.snapshot_download 下载模型到本地缓存;
2. 默认通过 HF_ENDPOINT 指定的镜像下载;
3. 只下载 Safetensors 权重及模型运行所需文件;
4. 排除 .bin/.pt/.pth/GGUF 和 original/ 原始权重;
5. 下载完成后使用本地快照进行加载校验;
6. 下载失败时自动重试,但不会继续执行加载校验.

示例:
    python download_models.py

只下载指定模型:
    python download_models.py \
        --model NousResearch/Meta-Llama-3.1-8B

指定多个模型:
    python download_models.py \
        --model Qwen/Qwen3-4B \
        --model NousResearch/Meta-Llama-3.1-8B

只下载,不执行模型加载校验:
    python download_models.py \
        --model NousResearch/Meta-Llama-3.1-8B \
        --skip-load-check
"""

import os

# ============================================================
# 必须在导入 huggingface_hub / transformers 之前设置环境变量
# ============================================================

# 允许在启动命令中通过环境变量覆盖镜像地址.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 元数据/ETag/HEAD 请求超时.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

# 单个文件下载阶段的超时时间.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

# 第三方镜像可能无法完整兼容 Hugging Face Xet 重定向.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# 不使用已经弃用的 hf_transfer 下载方式.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

# 如需查看详细 HTTP/cURL 调试日志,可在启动前设置:
# export HF_DEBUG=1
# os.environ.setdefault("HF_DEBUG", "1")


import argparse
import gc
import time
import traceback
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 默认模型列表
# ============================================================

MODELS = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "NousResearch/Meta-Llama-3.1-8B",
    "mistralai/Ministral-8B-Instruct-2410",
]


# ============================================================
# 下载文件规则
# ============================================================

# 使用白名单,只获取模型运行通常需要的文件.
#
# 相比 ignore_patterns,allow_patterns 更严格:
# 即使仓库以后增加了 .pth/GGUF 或其他大文件,也不会误下载.
ALLOW_PATTERNS = [
    # Transformers Safetensors 权重
    "*.safetensors",
    "*.safetensors.index.json",

    # 模型配置
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",

    # Tokenizer 配置和资源
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "vocab.txt",
    "*.model",
    "*.tiktoken",
    "*.jinja",
    "chat_template.json",
    "chat_template.jinja",

    # 部分模型使用自定义 Transformers 代码
    "*.py",

    # 仓库说明和许可证
    "README*",
    "LICENSE*",
    "NOTICE*",
    "USE_POLICY*",
    ".gitattributes",
]

# 双重保险:即使某个 allow pattern 意外匹配,也排除这些文件.
IGNORE_PATTERNS = [
    # 非 Safetensors 权重
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",

    # 其他框架权重
    "*.flax",
    "*.h5",
    "*.tflite",
    "*.msgpack",

    # 量化和推理格式
    "*.gguf",
    "*.ggml",
    "*.onnx",

    # Meta 原始格式权重目录
    "original/*",
    "original/**",

    # 通常不需要的训练产物
    "optimizer.pt",
    "scheduler.pt",
    "training_args.bin",
]


# ============================================================
# 工具函数
# ============================================================

def get_load_dtype() -> torch.dtype:
    """
    根据运行设备选择加载数据类型.

    CUDA 且支持 BF16:
        torch.bfloat16

    CUDA 但不支持 BF16:
        torch.float16

    无 CUDA:
        torch.float32
    """
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    return torch.float32


def clear_memory() -> None:
    """
    尽可能释放 Python/CPU 和 CUDA 缓存.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def check_safetensors_files(snapshot_path: str) -> list[Path]:
    """
    检查下载目录中是否包含 Safetensors 权重文件.
    """
    path = Path(snapshot_path)

    if not path.exists():
        raise FileNotFoundError(f"快照目录不存在:{snapshot_path}")

    files = sorted(path.rglob("*.safetensors"))

    if not files:
        raise FileNotFoundError(
            f"在快照目录中没有发现 .safetensors 权重:{snapshot_path}"
        )

    return files


# ============================================================
# 下载函数
# ============================================================

def robust_download(
    model_name: str,
    max_retries: int = 10,
    max_workers: int = 2,
) -> Optional[str]:
    """
    下载指定模型,并返回本地快照目录.

    下载成功:
        返回 snapshot_download 生成的本地快照路径.

    下载失败:
        返回 None.

    参数:
        model_name:
            Hugging Face 模型仓库名称.

        max_retries:
            最大尝试次数,包含第一次下载.

        max_workers:
            并发下载线程数量.第三方镜像建议设置为 1～4.
    """
    print(f"\n{'=' * 80}")
    print(f"[Download] 正在检查/下载:{model_name}")
    print(f"[Endpoint] {os.getenv('HF_ENDPOINT')}")
    print(f"[Retries] 最大尝试次数:{max_retries}")
    print(f"[Workers] 并发下载数:{max_workers}")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Download] 第 {attempt}/{max_retries} 次尝试")

            snapshot_path = snapshot_download(
                repo_id=model_name,
                allow_patterns=ALLOW_PATTERNS,
                ignore_patterns=IGNORE_PATTERNS,
                max_workers=max_workers,
            )

            safetensors_files = check_safetensors_files(snapshot_path)
            total_size = sum(file.stat().st_size for file in safetensors_files)

            print(f"[Download] ✅ {model_name} 准备就绪")
            print(f"[Cache] {snapshot_path}")
            print(f"[Safetensors] 分片数量:{len(safetensors_files)}")
            print(f"[Safetensors] 权重大小:{total_size / 1024**3:.2f} GiB")

            return snapshot_path

        except KeyboardInterrupt:
            print("\n[Interrupted] 用户终止下载.")
            raise

        except Exception as exc:
            print(
                f"[Warning] 第 {attempt}/{max_retries} 次下载失败:"
                f"{type(exc).__name__}: {exc}"
            )

            # 最后一次失败时输出完整堆栈,方便定位底层原因.
            if attempt >= max_retries:
                print(f"[Error] {model_name} 达到最大尝试次数.")
                traceback.print_exc()
                return None

            # 逐步增加等待时间,最多等待 30 秒.
            wait_seconds = min(5 * attempt, 30)
            print(f"[Retry] {wait_seconds} 秒后重新尝试^^")
            time.sleep(wait_seconds)

    return None


# ============================================================
# 模型加载校验
# ============================================================

def touch_model(
    model_name: str,
    snapshot_path: str,
    trust_remote_code: bool = True,
) -> bool:
    """
    从已经下载完成的本地快照加载模型,校验配置/Tokenizer 和权重.

    注意:
    - 直接使用 snapshot_path,而不是再次通过 model_name 查询 Hub;
    - local_files_only=True,确保加载阶段不访问网络;
    - use_safetensors=True,禁止回退到 .bin 权重.
    """
    print(f"\n[Loading] {model_name}")
    print(f"[Local snapshot] {snapshot_path}")

    dtype = get_load_dtype()
    print(f"[DType] {dtype}")
    print(f"[CUDA] {torch.cuda.is_available()}")

    try:
        config = AutoConfig.from_pretrained(
            snapshot_path,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

        load_kwargs = {
            "config": config,
            "local_files_only": True,
            "trust_remote_code": trust_remote_code,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
            "torch_dtype": dtype,
        }

        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
        else:
            # CPU 环境不使用 device_map="auto",避免不必要的 accelerate 依赖.
            load_kwargs["device_map"] = None

        model = AutoModelForCausalLM.from_pretrained(
            snapshot_path,
            **load_kwargs,
        )

        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        print(f"[Config] model_type={getattr(config, 'model_type', 'unknown')}")
        print(f"[Tokenizer] vocab_size={len(tokenizer):,}")
        print(f"[Parameters] {parameter_count:,}")
        print(
            f"[OK] 成功加载并校验:{model_name} "
            f"(Safetensors, {str(dtype).replace('torch.', '').upper()})"
        )

        del model
        del tokenizer
        del config
        clear_memory()

        return True

    except KeyboardInterrupt:
        print("\n[Interrupted] 用户终止模型加载.")
        clear_memory()
        raise

    except Exception as exc:
        print(
            f"[Error] 加载失败:{model_name}\n"
            f"[Exception] {type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        clear_memory()
        return False


# ============================================================
# 命令行参数
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量下载并校验 Hugging Face Safetensors 模型"
    )

    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help=(
            "只处理指定模型,可以重复传入."
            "未指定时处理脚本 MODELS 列表中的全部模型."
        ),
    )

    parser.add_argument(
        "--skip-load-check",
        action="store_true",
        help="只下载到本地缓存,不加载完整模型进行校验.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="每个模型的最大下载尝试次数,默认 10.",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="并发下载线程数,默认 2.镜像不稳定时可设置为 1.",
    )

    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="加载模型时禁用 trust_remote_code.",
    )

    return parser.parse_args()


# ============================================================
# 主程序
# ============================================================

def main() -> int:
    args = parse_args()

    if args.max_retries < 1:
        raise ValueError("--max-retries 必须大于或等于 1")

    if args.max_workers < 1:
        raise ValueError("--max-workers 必须大于或等于 1")

    models = args.models or MODELS
    trust_remote_code = not args.no_trust_remote_code

    print("=" * 80)
    print("Hugging Face 模型下载与加载校验")
    print("=" * 80)
    print(f"HF_HOME: {os.getenv('HF_HOME') or '未显式设置,使用默认缓存目录'}")
    print(f"HF_ENDPOINT: {os.getenv('HF_ENDPOINT')}")
    print(f"HF_HUB_ETAG_TIMEOUT: {os.getenv('HF_HUB_ETAG_TIMEOUT')}")
    print(f"HF_HUB_DOWNLOAD_TIMEOUT: {os.getenv('HF_HUB_DOWNLOAD_TIMEOUT')}")
    print(f"HF_HUB_DISABLE_XET: {os.getenv('HF_HUB_DISABLE_XET')}")
    print(f"trust_remote_code: {trust_remote_code}")
    print(f"待处理模型数量: {len(models)}")

    download_success = 0
    load_success = 0
    failed_models: list[str] = []

    for model_name in models:
        snapshot_path = robust_download(
            model_name=model_name,
            max_retries=args.max_retries,
            max_workers=args.max_workers,
        )

        # 下载失败后不执行加载.
        if snapshot_path is None:
            failed_models.append(model_name)
            continue

        download_success += 1

        if args.skip_load_check:
            continue

        loaded = touch_model(
            model_name=model_name,
            snapshot_path=snapshot_path,
            trust_remote_code=trust_remote_code,
        )

        if loaded:
            load_success += 1
        else:
            failed_models.append(model_name)

    print(f"\n{'=' * 80}")
    print("[Summary]")
    print(f"模型总数:{len(models)}")
    print(f"下载成功:{download_success}")

    if args.skip_load_check:
        print("加载校验:已跳过")
    else:
        print(f"加载成功:{load_success}")

    if failed_models:
        print("[Failed models]")
        for model_name in failed_models:
            print(f"  - {model_name}")
        return 1

    print("[Done] ✅ 所有模型处理完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
