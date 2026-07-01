import sys
import os
import argparse
import logging
import csv
import datetime

# 将项目根目录添加到系统路径,以确保能够正确导入 utils 和 models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import PackKVCacheConfig
from utils.compute import QuantMethod, RepackMethod
from evaluation.evaluation import cr_evaluation
from utils.util import get_logger, block_other_logger

max_ctx_len_map = {
    "Qwen/Qwen3-4B": 1024 * 40,  # 40K 上下文
    "Qwen/Qwen3-8B": 1024 * 40,  # 40K 上下文
    "NousResearch/Meta-Llama-3-8B": 1024 * 8,  # 8K 上下文 (区别于 3.1 版本的 128K)
    "JackFram/llama-160m": 1024 * 2,  # 2K 上下文
}


def append_to_macro_summary_csv(
    args, avg_k_cr, avg_v_cr, overall_avg, k_save_pct, v_save_pct, csv_path
):
    """
    将全局宏观结果追加 (Append) 到一个总的 CSV 汇总表中
    """
    save_dir = "./csv_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    summary_file = os.path.join(save_dir, "Global_Macro_Summary.csv")
    file_exists = os.path.isfile(summary_file)

    # 定义表头 (涵盖了你对比实验需要的所有超参和结果)
    headers = [
        "Timestamp",
        "Model",
        "Ctx_Len",
        "Quant_Method",
        "Repack_Method",
        "K_Scale",
        "V_Scale",
        "Block_Size",
        "Pack_Size",
        "K_Avg_CR",
        "V_Avg_CR",
        "Overall_Avg_CR",
        "K_Mem_Saved(%)",
        "V_Mem_Saved(%)",
        "csv_path",
    ]

    # 组装当前运行的数据行
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_data = [
        timestamp,
        args.model_name,
        args.ctx_len,
        args.quant_method,
        args.repack_method,
        args.k_scale,
        args.v_scale,
        args.block_size,
        args.pack_size,
        f"{avg_k_cr:.4f}",
        f"{avg_v_cr:.4f}",
        f"{overall_avg:.4f}",
        f"{k_save_pct:.2f}%",
        f"{v_save_pct:.2f}%",
        f"{csv_path}",
    ]

    try:
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

    # 生成安全的文件名
    safe_model_name = args.model_name.replace("/", "_").replace("\\", "_")
    timestamp = datetime.datetime.now().strftime("%Y%md_%H%M%S")
    csv_filename = f"CR_Report_{safe_model_name}_ctx{args.ctx_len}_Round{round_idx}_{timestamp}.csv"
    csv_path = os.path.join(save_dir, csv_filename)

    # 提取共有多少层 (以 k_original_size 的长度为准)
    num_layers = 0
    if "k_original_size" in res_dict and isinstance(res_dict["k_original_size"], list):
        num_layers = len(res_dict["k_original_size"])

    if num_layers == 0:
        return None  # 如果没有逐层数据,跳过导出

    # 准备 CSV 表头
    headers = ["Layer"]
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
                row = [f"Layer_{layer_idx}"]
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
    # parser.add_argument(
    #     "-c",
    #     "--ctx_len",
    #     type=int,
    #     default=2048,
    #     help="用于提取高精度缓存的上下文长度 (Context Length)",
    # )

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

    try:
        quant_method_enum = QuantMethod[args.quant_method]
        repack_method_enum = RepackMethod[args.repack_method]
    except KeyError as e:
        logger.error(
            f"错误的枚举参数: {e}. 请检查 --quant_method 或 --repack_method 的拼写."
        )
        sys.exit(1)

    config = PackKVCacheConfig(
        enable_quant=False,
        model_name=args.model_name,
        quant_method=quant_method_enum,
        repack_method=repack_method_enum,
        high_precision_zero_point=True,  # 默认开启高精度极值保存
        block_size=args.block_size,
        buffer_size=args.buffer_size,
        pack_size=args.pack_size,
        k_quant_scale_rel=args.k_scale,
        v_quant_scale_rel=args.v_scale,
    )
    args.ctx_len = max_ctx_len_map[args.model_name]
    args.enable_save = True
    logger.info("=" * 50)
    logger.info(f"   开始压缩率 (CR) 评测: {args.model_name}")
    logger.info(f"   Context Length: {args.ctx_len}")
    logger.info(f"   K Scale: {args.k_scale}, V Scale: {args.v_scale}")
    logger.info(f"   Block Size: {args.block_size}, Pack Size: {args.pack_size}")
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
                    # 计算所有层的平均压缩率
                    k_cr_list = res["k_encode_after_repack_cr"]
                    v_cr_list = res["v_encode_after_repack_cr"]

                    avg_k_cr = sum(k_cr_list) / len(k_cr_list) if k_cr_list else 0
                    avg_v_cr = sum(v_cr_list) / len(v_cr_list) if v_cr_list else 0

                    k_save_pct = (1.0 - 1.0 / avg_k_cr) * 100 if avg_k_cr > 0 else 0
                    v_save_pct = (1.0 - 1.0 / avg_v_cr) * 100 if avg_v_cr > 0 else 0

                    print(
                        f"   Key Cache 平均压缩率   : {avg_k_cr:.3f}x  (显存节省: {k_save_pct:.1f}%)"
                    )
                    print(
                        f"   Value Cache 平均压缩率 : {avg_v_cr:.3f}x  (显存节省: {v_save_pct:.1f}%)"
                    )
                    print("-" * 50)
                    overall_avg = (avg_k_cr + avg_v_cr) / 2
                    print(f"   综合全局平均压缩率     : {overall_avg:.3f}x")

                    # 导出详细数据到 CSV
                    csv_path = export_to_csv(args, res, i + 1)
                    if csv_path:
                        print(f"   逐层详细数据已导出至 : {csv_path}")

                    summary_path = append_to_macro_summary_csv(
                        args,
                        avg_k_cr,
                        avg_v_cr,
                        overall_avg,
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
