#! /usr/bin/env python
import sys
import os
import argparse
import shutil

# 允许从仓库根目录直接执行 scripts/accuracy/accuracy_eval.py。
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from evaluation.evaluation import accuracy_evaluation
from utils.compute import (
    BucketScoreMethod,
    QuantMethod,
    RepackMethod,
    ScaleMethod,
)
from utils.config import PackKVCacheConfig
from utils.util import get_logger, block_other_logger

logger = get_logger(__file__)
block_other_logger(logger)


def parse_arguments():
    parser = argparse.ArgumentParser(description="PackKV仿真评测CLI工具")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="指定大模型 Huggingface 路径或名称",
    )
    parser.add_argument(
        "-t",
        "--tasks",
        type=str,
        required=True,
        help="指定评测数据集任务名称 (单个任务)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="./eval_logs", help="输出文件夹的根路径"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=str,
        default="auto",
        help="Batch size 可填整数或 'auto'",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Debug参数:限制每个任务只跑前 N 道题",
    )

    # PackKV 专属量化控制参数(方便你后续切换测试量化对齐差距)
    parser.add_argument(
        "--quant_method",
        type=str,
        default="PackKV",
        choices=["PackKV", "KIVI"],
        help="量化算法选择",
    )
    parser.add_argument(
        "--k_scale", type=float, default=0.1, help="K Cache 相对缩放因子 scale_rel"
    )
    parser.add_argument(
        "--v_scale", type=float, default=0.1, help="V Cache 相对缩放因子 scale_rel"
    )
    parser.add_argument(
        "--scale_method",
        type=str,
        default=ScaleMethod.CONTINUOUS.value,
        choices=[method.value for method in ScaleMethod],
        help="量化步长策略",
    )
    parser.add_argument(
        "--high_precision_zero_point",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="保存浮点 minimum；默认沿用原精度脚本的整数 zero point",
    )
    parser.add_argument("--k_error_budget", type=float, default=0.1)
    parser.add_argument("--v_error_budget", type=float, default=0.1)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=192)
    parser.add_argument("--pack_size", type=int, default=16)
    parser.add_argument("--bucket_count", type=int, default=4)
    parser.add_argument(
        "--bucket_score_method",
        choices=[method.value for method in BucketScoreMethod],
        default=BucketScoreMethod.K_SUM.value,
    )
    parser.add_argument(
        "--no_quant",
        action="store_true",
        help="关闭量化，生成全精度基线",
    )
    parser.add_argument(
        "--clear_lm_eval_cache",
        action="store_true",
        help="运行前删除 ~/.cache/lm-eval；默认不删除",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.k_error_budget < 0 or args.v_error_budget < 0:
        raise SystemExit("K/V error budget must be non-negative")
    if args.block_size <= 0 or args.pack_size <= 0 or args.buffer_size < 0:
        raise SystemExit("block_size/pack_size must be positive and buffer_size non-negative")
    if args.block_size % args.pack_size:
        raise SystemExit("block_size must be divisible by pack_size")
    if args.scale_method == ScaleMethod.PO2_PACK_AWARE.value and (
        args.quant_method != "PackKV"
        or args.bucket_score_method != BucketScoreMethod.K_SUM.value
    ):
        raise SystemExit(
            "po2_pack_aware requires --quant_method PackKV "
            "--bucket_score_method k_sum"
        )
    if args.scale_method == ScaleMethod.PO2_PACK_AWARE.value and (
        args.bucket_count < 2
        or args.bucket_count > args.block_size
        or args.bucket_count & (args.bucket_count - 1)
    ):
        raise SystemExit("bucket_count must be a power of two in [2, block_size]")

    # 删除缓存是显式可选的破坏性操作；默认保留 lm-eval 缓存。
    cache_dir = os.path.expanduser("~/.cache/lm-eval")
    if args.clear_lm_eval_cache and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info("已清除 lm-eval 缓存.")

    # 处理 batch_size 参数类型转换
    try:
        final_batch_size = int(args.batch_size)
    except ValueError:
        final_batch_size = args.batch_size  # 如果传的是 "auto" 则保持字符串
    if (
        args.scale_method == ScaleMethod.PO2_PACK_AWARE.value
        and final_batch_size != 1
    ):
        raise SystemExit(
            "po2_pack_aware 精度参考路径当前要求 --batch_size 1，"
            "避免不同样本共享同一 pack 选择掩码"
        )

    # 选择量化算法枚举
    chosen_method = (
        QuantMethod.PackKV if args.quant_method == "PackKV" else QuantMethod.KIVI
    )

    # 组装 PackKV 算法控制配方
    config = PackKVCacheConfig(
        enable_quant=not args.no_quant,
        model_name=args.model,
        quant_method=chosen_method,
        repack_method=(
            RepackMethod.BUCKET
            if args.scale_method == ScaleMethod.PO2_PACK_AWARE.value
            else RepackMethod.NONE
        ),
        high_precision_zero_point=args.high_precision_zero_point,
        block_size=args.block_size,
        buffer_size=args.buffer_size,
        pack_size=args.pack_size,
        k_quant_scale_rel=args.k_scale,
        v_quant_scale_rel=args.v_scale,
        scale_method=ScaleMethod(args.scale_method),
        bucket_count=args.bucket_count,
        bucket_score_method=BucketScoreMethod(args.bucket_score_method),
        k_error_budget=args.k_error_budget,
        v_error_budget=args.v_error_budget,
    )

    logger.info("========================================")
    logger.info(f"  模型: {args.model}")
    logger.info(f"  任务: {args.tasks}")
    logger.info(
        f"  量化方案: {args.quant_method} (K_scale={args.k_scale}, V_scale={args.v_scale})"
    )
    logger.info(f"  Scale 策略: {args.scale_method}")
    logger.info(f"  启用量化: {not args.no_quant}")
    logger.info(
        f"  Layer SSE 预算: K={args.k_error_budget}, V={args.v_error_budget}"
    )
    logger.info(f"  高精度零点: {args.high_precision_zero_point}")
    logger.info("========================================")

    _ = accuracy_evaluation(
        config=config,
        benchmark=args.tasks,
        logger=logger,
        batch_size=final_batch_size,
        limit=args.limit,
        output_dir=args.output,
    )

    logger.info("本轮指令运行完毕.")


if __name__ == "__main__":
    main()
