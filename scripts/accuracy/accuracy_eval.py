#! /usr/bin/env python
import sys
import os
import argparse
import shutil
from evaluation.evaluation import accuracy_evaluation
from utils.compute import QuantMethod, RepackMethod
from utils.config import PackKVCacheConfig
from utils.util import get_logger, block_other_logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

    return parser.parse_args()


def main():
    args = parse_arguments()

    # 清楚lm_eval内存缓存, 避免 lm_eval偷懒返回错误的旧数据
    cache_dir = os.path.expanduser("~/.cache/lm-eval")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info("已清楚缓存.")

    # 处理 batch_size 参数类型转换
    try:
        final_batch_size = int(args.batch_size)
    except ValueError:
        final_batch_size = args.batch_size  # 如果传的是 "auto" 则保持字符串

    # 选择量化算法枚举
    chosen_method = (
        QuantMethod.PackKV if args.quant_method == "PackKV" else QuantMethod.KIVI
    )

    # 组装 PackKV 算法控制配方
    config = PackKVCacheConfig(
        enable_quant=True,
        model_name=args.model,
        quant_method=chosen_method,
        repack_method=RepackMethod.NONE,
        high_precision_zero_point=False,
        block_size=64,
        buffer_size=128 + 64,
        pack_size=16,
        k_quant_scale_rel=args.k_scale,
        v_quant_scale_rel=args.v_scale,
    )

    logger.info("========================================")
    logger.info(f"  模型: {args.model}")
    logger.info(f"  任务: {args.tasks}")
    logger.info(
        f"  量化方案: {args.quant_method} (K_scale={args.k_scale}, V_scale={args.v_scale})"
    )
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
