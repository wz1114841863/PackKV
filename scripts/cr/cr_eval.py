import sys
import os
import argparse
import logging
import csv
import datetime
import re

# 将项目根目录添加到系统路径,确保可直接运行 scripts/cr/cr_eval.py.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from utils.config import PackKVCacheConfig
from utils.compute import (
    BucketScoreMethod,
    QuantMethod,
    RepackMethod,
    ScaleMethod,
)
from evaluation.evaluation import cr_evaluation
from utils.util import get_logger, block_other_logger

max_ctx_len_map = {
    "Qwen/Qwen3-4B": 1024 * 4,
    "Qwen/Qwen3-8B": 1024 * 4,
    "NousResearch/Meta-Llama-3-8B": 1024 * 4,  # 8K 上下文 (区别于 3.1 版本的 128K)
    "mistralai/Ministral-8B-Instruct-2410": 1024 * 4,
    "JackFram/llama-160m": 1024 * 2,  # 2K 上下文
}

STORAGE_MODEL = "stream-packed-v2-native-quant-metadata-bucket-counts"


def safe_filename_component(value):
    """将实验参数转换为适合文件名的稳定片段."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")


def build_report_filename(args, round_idx, timestamp=None):
    """生成包含完整实验配置的逐层报告文件名."""
    timestamp = timestamp or datetime.datetime.now()
    zero_point_mode = (
        "fp-min" if args.high_precision_zero_point else "int-zero"
    )
    fields = [
        "CR",
        safe_filename_component(args.model_name),
        f"ctx-{args.ctx_len}",
        f"quant-{args.quant_method}",
        f"scale-{args.scale_method}",
        f"zp-{zero_point_mode}",
        f"repack-{args.repack_method}",
        f"k-{safe_filename_component(args.k_scale)}",
        f"v-{safe_filename_component(args.v_scale)}",
        f"block-{args.block_size}",
        f"buffer-{args.buffer_size}",
        f"pack-{args.pack_size}",
        f"buckets-{args.bucket_count}",
        f"bucket-score-{args.bucket_score_method}",
        f"round-{round_idx}",
        timestamp.strftime("%Y%m%d_%H%M%S"),
    ]
    return "_".join(fields) + ".csv"


def append_to_macro_summary_csv(
    args,
    k_original_bytes,
    v_original_bytes,
    k_compressed_bytes,
    v_compressed_bytes,
    k_global_cr,
    v_global_cr,
    overall_global_cr,
    k_save_pct,
    v_save_pct,
    csv_path,
):
    """
    将全局宏观结果追加 (Append) 到一个总的 CSV 汇总表中
    """
    save_dir = "./csv_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # v5 增加 Bucket score 方法和 K/V 二级桶配置.
    summary_file = os.path.join(save_dir, "Global_Macro_Summary_v5.csv")
    file_exists = os.path.isfile(summary_file)

    # 定义表头 (涵盖了你对比实验需要的所有超参和结果)
    headers = [
        "Timestamp",
        "Model",
        "Ctx_Len",
        "Quant_Method",
        "Scale_Method",
        "High_Precision_Zero_Point",
        "Repack_Method",
        "K_Scale",
        "V_Scale",
        "Block_Size",
        "Buffer_Size",
        "Pack_Size",
        "Bucket_Count",
        "Bucket_Score_Method",
        "Storage_Model",
        "K_Original_Bytes",
        "V_Original_Bytes",
        "K_Compressed_Bytes",
        "V_Compressed_Bytes",
        "K_Global_CR",
        "V_Global_CR",
        "Overall_Global_CR",
        "K_Mem_Saved(%)",
        "V_Mem_Saved(%)",
        "Detailed_Report_Path",
    ]

    # 组装当前运行的数据行
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_data = [
        timestamp,
        args.model_name,
        args.ctx_len,
        args.quant_method,
        args.scale_method,
        args.high_precision_zero_point,
        args.repack_method,
        args.k_scale,
        args.v_scale,
        args.block_size,
        args.buffer_size,
        args.pack_size,
        args.bucket_count,
        args.bucket_score_method,
        STORAGE_MODEL,
        k_original_bytes,
        v_original_bytes,
        k_compressed_bytes,
        v_compressed_bytes,
        f"{k_global_cr:.4f}",
        f"{v_global_cr:.4f}",
        f"{overall_global_cr:.4f}",
        f"{k_save_pct:.2f}%",
        f"{v_save_pct:.2f}%",
        f"{csv_path}",
    ]

    try:
        if file_exists:
            with open(summary_file, mode="r", newline="", encoding="utf-8") as f:
                existing_headers = next(csv.reader(f), None)
            if existing_headers != headers:
                raise ValueError(
                    "Global_Macro_Summary_v5.csv 表头与当前 schema 不一致"
                )

        # 使用 'a' 模式追加写入
        with open(summary_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 如果文件是新建的,先写一行表头
            if not file_exists:
                writer.writerow(headers)
            # 写入当前跑完的这一行数据
            writer.writerow(row_data)
        return summary_file
    except Exception as e:
        print(f" 写入宏观汇总表失败: {e}")
        return None


def export_to_csv(args, res_dict, round_idx):
    """
    将详细的逐层 (Layer-wise) 评测数据导出为 CSV 文件
    """
    # 创建保存目录
    save_dir = "./csv_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    generated_at = datetime.datetime.now()
    csv_filename = build_report_filename(args, round_idx, generated_at)
    csv_path = os.path.join(save_dir, csv_filename)

    # 提取共有多少层 (以 k_original_size 的长度为准)
    num_layers = 0
    if "k_original_size" in res_dict and isinstance(res_dict["k_original_size"], list):
        num_layers = len(res_dict["k_original_size"])

    if num_layers == 0:
        return None  # 如果没有逐层数据,跳过导出

    # 准备 CSV 表头
    metadata_headers = [
        "Generated_At",
        "Model",
        "Ctx_Len",
        "Quant_Method",
        "Scale_Method",
        "High_Precision_Zero_Point",
        "Repack_Method",
        "K_Scale",
        "V_Scale",
        "Block_Size",
        "Buffer_Size",
        "Pack_Size",
        "Bucket_Count",
        "Bucket_Score_Method",
        "Round",
        "Storage_Model",
    ]
    metadata_values = [
        generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        args.model_name,
        args.ctx_len,
        args.quant_method,
        args.scale_method,
        args.high_precision_zero_point,
        args.repack_method,
        args.k_scale,
        args.v_scale,
        args.block_size,
        args.buffer_size,
        args.pack_size,
        args.bucket_count,
        args.bucket_score_method,
        round_idx,
        STORAGE_MODEL,
    ]
    headers = metadata_headers + ["Layer"]
    # 提取所有值为列表的键作为列名
    list_keys = [
        k for k, v in res_dict.items() if isinstance(v, list) and len(v) == num_layers
    ]
    headers.extend(list_keys)

    try:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            # 逐行 (逐层) 写入数据
            for layer_idx in range(num_layers):
                row = metadata_values + [f"Layer_{layer_idx}"]
                for key in list_keys:
                    row.append(res_dict[key][layer_idx])
                writer.writerow(row)
        return csv_path
    except Exception as e:
        print(f"❌ 导出 CSV 失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="PackKV 压缩率评测")

    # 基础模型与上下文参数
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="meta-llama/Llama-3-8B",
        help="需要评测的模型名称或本地路径",
    )
    parser.add_argument(
        "-c",
        "--ctx_len",
        type=int,
        default=None,
        help=(
            "用于提取高精度缓存的上下文长度;未指定时使用已知模型的默认值. "
            "本地路径或未登记模型必须显式指定"
        ),
    )

    # PackKV 核心算法超参
    parser.add_argument(
        "--block_size", type=int, default=64, help="量化切块大小 (Block Size)"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=128 + 64,
        help="保留的高精度缓存大小 (Buffer Size)",
    )
    parser.add_argument(
        "--pack_size", type=int, default=16, help="位宽重排打包的对齐大小 (Pack Size)"
    )
    parser.add_argument(
        "--bucket_count",
        type=int,
        default=4,
        help="BUCKET 重排的 FIFO 桶数;必须是不超过 block_size 的 2 的幂",
    )
    parser.add_argument(
        "--bucket_score_method",
        type=str,
        default=BucketScoreMethod.COMBINED_SUM.value,
        choices=[method.value for method in BucketScoreMethod],
        help="BUCKET 的整数 score:当前基线/K-only/V-only/K-V二级分桶",
    )

    # 量化精度控制参数
    parser.add_argument(
        "--k_scale",
        type=float,
        default=0.01,
        help="K Cache 量化的相对误差容忍度 (Scale Rel)",
    )
    parser.add_argument(
        "--v_scale",
        type=float,
        default=0.01,
        help="V Cache 量化的相对误差容忍度 (Scale Rel)",
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
        default=True,
        help="保存浮点 minimum;使用 --no-high_precision_zero_point 改为整数 zero point",
    )

    # 方法选项
    parser.add_argument(
        "--quant_method",
        type=str,
        default="PackKV",
        choices=["KIVI", "PackKV"],
        help="量化方法枚举名称",
    )
    parser.add_argument(
        "--repack_method",
        type=str,
        default="GREEDY",
        choices=["GREEDY", "MEDIAN", "BUCKET", "NONE"],
        help="编码感知重排算法策略",
    )

    # 缓存提取与存储相关
    # parser.add_argument(
    #     "--enable_save",
    #     action="store_true",
    #     help="是否将提取的高精度 Cache 保存到磁盘 (触发缓存命中机制)",
    # )
    parser.add_argument("--collect_round", type=int, default=1, help="提取数据的轮数")

    args = parser.parse_args()

    logger = get_logger(__name__)
    block_other_logger(logger)
    logger.setLevel(logging.INFO)

    if args.ctx_len is None:
        if args.model_name not in max_ctx_len_map:
            parser.error(
                f"模型 {args.model_name!r} 没有默认上下文长度;请显式传入 --ctx_len"
            )
        args.ctx_len = max_ctx_len_map[args.model_name]
    if args.ctx_len <= 0:
        parser.error("--ctx_len must be a positive integer")

    try:
        quant_method_enum = QuantMethod[args.quant_method]
        repack_method_enum = RepackMethod[args.repack_method]
    except KeyError as e:
        logger.error(
            f"错误的枚举参数: {e}. 请检查 --quant_method 或 --repack_method 的拼写."
        )
        sys.exit(1)
    if repack_method_enum == RepackMethod.BUCKET and (
        args.bucket_count < 2
        or args.bucket_count > args.block_size
        or args.bucket_count & (args.bucket_count - 1)
    ):
        parser.error(
            "--bucket_count must be a power of two in [2, block_size] "
            "when --repack_method BUCKET"
        )
    if (
        repack_method_enum == RepackMethod.BUCKET
        and args.bucket_score_method == BucketScoreMethod.KV_2D.value
        and args.bucket_count < 4
    ):
        parser.error("--bucket_score_method kv_2d requires --bucket_count >= 4")

    config = PackKVCacheConfig(
        enable_quant=False,
        model_name=args.model_name,
        quant_method=quant_method_enum,
        repack_method=repack_method_enum,
        high_precision_zero_point=args.high_precision_zero_point,
        block_size=args.block_size,
        buffer_size=args.buffer_size,
        pack_size=args.pack_size,
        k_quant_scale_rel=args.k_scale,
        v_quant_scale_rel=args.v_scale,
        scale_method=ScaleMethod(args.scale_method),
        bucket_count=args.bucket_count,
        bucket_score_method=BucketScoreMethod(args.bucket_score_method),
    )
    args.enable_save = True
    logger.info("=" * 50)
    logger.info(f"   开始压缩率 (CR) 评测: {args.model_name}")
    logger.info(f"   Context Length: {args.ctx_len}")
    logger.info(f"   K Scale: {args.k_scale}, V Scale: {args.v_scale}")
    logger.info(f"   Scale Method: {args.scale_method}")
    logger.info(f"   High Precision Zero Point: {args.high_precision_zero_point}")
    logger.info(f"   Block Size: {args.block_size}, Pack Size: {args.pack_size}")
    logger.info(f"   Bucket Count: {args.bucket_count}")
    logger.info(f"   Bucket Score Method: {args.bucket_score_method}")
    logger.info("=" * 50)

    results = cr_evaluation(
        config=config,
        ctx_len=args.ctx_len,
        enable_save=args.enable_save,
        logger=logger,
        collect_round=args.collect_round,
    )

    # 打印最终结果
    print("\n" + "=" * 20)
    print(f"    [PackKV 压缩率 (CR) 宏观报告]")
    print(f"    模型: {args.model_name} | Ctx: {args.ctx_len}")
    print("=" * 20)

    if not results:
        print("未返回任何结果.")
    else:
        for i, res in enumerate(results):
            print(f"\n [Round {i+1}]")

            if isinstance(res, dict):
                # 如果字典中包含我们需要的逐层压缩率数组 (例如 'k_encode_after_repack_cr')
                if (
                    "k_encode_after_repack_cr" in res
                    and "v_encode_after_repack_cr" in res
                ):
                    k_original_bytes = sum(res["k_original_size"])
                    v_original_bytes = sum(res["v_original_size"])
                    k_compressed_bytes = sum(res["k_encode_size_after_repack"])
                    v_compressed_bytes = sum(res["v_encode_size_after_repack"])
                    k_global_cr = k_original_bytes / k_compressed_bytes
                    v_global_cr = v_original_bytes / v_compressed_bytes
                    overall_global_cr = (
                        k_original_bytes + v_original_bytes
                    ) / (k_compressed_bytes + v_compressed_bytes)
                    k_save_pct = (1.0 - 1.0 / k_global_cr) * 100
                    v_save_pct = (1.0 - 1.0 / v_global_cr) * 100

                    print(
                        f"   Key Cache 全局压缩率   : {k_global_cr:.3f}x  (显存节省: {k_save_pct:.1f}%)"
                    )
                    print(
                        f"   Value Cache 全局压缩率 : {v_global_cr:.3f}x  (显存节省: {v_save_pct:.1f}%)"
                    )
                    print("-" * 50)
                    print(f"   K/V 综合全局压缩率     : {overall_global_cr:.3f}x")
                    print(
                        "   统计口径: 总原始字节 / 总编码字节 "
                        "(包含Recent Buffer及量化元数据)"
                    )
                    print(f"   存储模型: {STORAGE_MODEL}")

                    # 导出详细数据到 CSV
                    csv_path = export_to_csv(args, res, i + 1)
                    if csv_path:
                        print(f"   逐层详细数据已导出至 : {csv_path}")

                    summary_path = append_to_macro_summary_csv(
                        args,
                        k_original_bytes,
                        v_original_bytes,
                        k_compressed_bytes,
                        v_compressed_bytes,
                        k_global_cr,
                        v_global_cr,
                        overall_global_cr,
                        k_save_pct,
                        v_save_pct,
                        csv_path,
                    )
                    if summary_path:
                        print(f"   宏观结果已追加至 : {summary_path}")

                else:
                    # 兼容其他格式的字典
                    print("   (未检测到标准的逐层数组,打印原始字典数据)")
                    for k, v in res.items():
                        if not isinstance(v, list):  # 只打印非数组的宏观值
                            print(f"   {k}: {v}")
            else:
                print(f"   原始数据: {res}")

    print("\n" + "=" * 20 + "\n")


if __name__ == "__main__":
    main()
