import sys
import os
import argparse
import logging
import csv
import datetime
import json
import statistics
import re
import uuid

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
PA_SELECTION_POLICY = "layer-budget-efficiency-prefix"


def sum_metric(res_dict, key):
    """汇总逐层数值指标;缺失指标按0处理以兼容旧评测路径."""
    values = res_dict.get(key, [])
    if not isinstance(values, list):
        return 0
    return sum(value for value in values if isinstance(value, (int, float)))


def aggregate_json_histogram(res_dict, key):
    """合并逐层 JSON histogram,返回稳定排序的字典."""
    merged = {}
    for raw_histogram in res_dict.get(key, []):
        if not raw_histogram:
            continue
        histogram = (
            json.loads(raw_histogram)
            if isinstance(raw_histogram, str)
            else raw_histogram
        )
        for bucket, count in histogram.items():
            bucket = str(bucket)
            merged[bucket] = merged.get(bucket, 0) + int(count)
    return dict(sorted(merged.items(), key=lambda item: int(item[0])))


def layer_statistics(res_dict, key):
    """返回逐层指标的 min/mean/max/population-std."""
    values = [
        float(value)
        for value in res_dict.get(key, [])
        if isinstance(value, (int, float))
    ]
    if not values:
        return "", "", "", ""
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return min(values), statistics.fmean(values), max(values), std


def format_float_or_blank(value, digits=8):
    return "" if value == "" else f"{value:.{digits}f}"


def result_metadata(args, round_idx, generated_at):
    """生成宏观表和统一逐层表共享的运行身份与实验配置."""
    sample_id = args.sample_id if args.sample_id is not None else round_idx - 1
    headers = [
        "Generated_At",
        "Suite_ID",
        "Run_ID",
        "Sample_ID",
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
        "K_Error_Budget",
        "V_Error_Budget",
        "PA_Selection_Policy",
        "Round",
        "Storage_Model",
    ]
    values = [
        generated_at.strftime("%Y-%m-%d %H:%M:%S.%f"),
        args.suite_id,
        args.run_id,
        sample_id,
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
        args.k_error_budget,
        args.v_error_budget,
        (
            PA_SELECTION_POLICY
            if args.scale_method == ScaleMethod.PO2_PACK_AWARE.value
            else ""
        ),
        round_idx,
        STORAGE_MODEL,
    ]
    return headers, values


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
    if args.scale_method == ScaleMethod.PO2_PACK_AWARE.value:
        fields.insert(-2, f"err-{args.k_error_budget}-{args.v_error_budget}")
        fields.insert(-2, "pa-layer-budget")
    return "_".join(fields) + ".csv"


def append_to_macro_summary_csv(
    args,
    res_dict,
    round_idx,
    csv_path,
    layer_detail_path,
):
    """将全局结果和逐层全局误差预算诊断追加到 v9 汇总表."""
    save_dir = "./csv_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    summary_file = os.path.join(save_dir, "Global_Macro_Summary_v9.csv")
    file_exists = os.path.isfile(summary_file)

    generated_at = datetime.datetime.now()
    metadata_headers, metadata_values = result_metadata(
        args, round_idx, generated_at
    )
    k_original_bytes = sum_metric(res_dict, "k_original_size")
    v_original_bytes = sum_metric(res_dict, "v_original_size")
    k_quant_bytes = sum_metric(res_dict, "k_quant_size")
    v_quant_bytes = sum_metric(res_dict, "v_quant_size")
    k_before_repack_bytes = sum_metric(
        res_dict, "k_encode_size_before_repack"
    )
    v_before_repack_bytes = sum_metric(
        res_dict, "v_encode_size_before_repack"
    )
    k_compressed_bytes = sum_metric(res_dict, "k_encode_size_after_repack")
    v_compressed_bytes = sum_metric(res_dict, "v_encode_size_after_repack")
    k_global_cr = k_original_bytes / k_compressed_bytes
    v_global_cr = v_original_bytes / v_compressed_bytes
    overall_global_cr = (k_original_bytes + v_original_bytes) / (
        k_compressed_bytes + v_compressed_bytes
    )
    k_save_pct = (1.0 - 1.0 / k_global_cr) * 100
    v_save_pct = (1.0 - 1.0 / v_global_cr) * 100

    component_suffixes = [
        "recent_high_precision_size",
        "quant_zero_point_size",
        "quant_scale_size",
        "bitpack_payload_size_after_repack",
        "bitpack_min_size_after_repack",
        "bitpack_encode_len_size_after_repack",
        "permutation_metadata_size",
        "bucket_metadata_size",
        "block_metadata_size",
    ]
    components = {
        cache_kind: {
            suffix: sum_metric(res_dict, f"{cache_kind}_{suffix}")
            for suffix in component_suffixes
        }
        for cache_kind in ("k", "v")
    }
    accounting_totals = {
        cache_kind: sum(components[cache_kind].values())
        for cache_kind in ("k", "v")
    }
    k_layer_stats = layer_statistics(res_dict, "k_encode_after_repack_cr")
    v_layer_stats = layer_statistics(res_dict, "v_encode_after_repack_cr")
    k_width_hist = aggregate_json_histogram(
        res_dict, "k_bit_width_hist_after_repack"
    )
    v_width_hist = aggregate_json_histogram(
        res_dict, "v_bit_width_hist_after_repack"
    )
    k_width_hist_before = aggregate_json_histogram(
        res_dict, "k_bit_width_hist_before_repack"
    )
    v_width_hist_before = aggregate_json_histogram(
        res_dict, "v_bit_width_hist_before_repack"
    )
    bucket_occupancy_hist = aggregate_json_histogram(
        res_dict, "repack_bucket_occupancy_hist"
    )
    bucket_slot_count = sum(bucket_occupancy_hist.values())
    bucket_empty_rate = (
        bucket_occupancy_hist.get("0", 0) / bucket_slot_count
        if bucket_slot_count
        else ""
    )
    pa_values = []
    for cache_kind in ("k", "v"):
        total_blocks = sum_metric(res_dict, f"{cache_kind}_pa_total_blocks")
        total_packs = sum_metric(res_dict, f"{cache_kind}_pa_total_packs")
        payload_beneficial = sum_metric(
            res_dict, f"{cache_kind}_pa_payload_beneficial_packs"
        )
        selected_packs = sum_metric(res_dict, f"{cache_kind}_pa_ceil_selected_packs")
        nearest_bits = sum_metric(res_dict, f"{cache_kind}_pa_nearest_payload_bits")
        ceil_bits = sum_metric(res_dict, f"{cache_kind}_pa_ceil_payload_bits")
        potential_bits = sum_metric(
            res_dict, f"{cache_kind}_pa_payload_benefit_ceiling_bits"
        )
        selected_bits = sum_metric(res_dict, f"{cache_kind}_pa_selected_payload_bits")
        nearest_sse = sum_metric(res_dict, f"{cache_kind}_pa_nearest_sse")
        ceil_sse = sum_metric(res_dict, f"{cache_kind}_pa_ceil_sse")
        selected_sse = sum_metric(res_dict, f"{cache_kind}_pa_selected_sse")
        error_budget_sse = sum_metric(res_dict, f"{cache_kind}_pa_error_budget_sse")
        used_delta_sse = selected_sse - nearest_sse
        layer_packs = res_dict.get(f"{cache_kind}_pa_total_packs", [])
        def weighted_mean(metric):
            values = res_dict.get(metric, [])
            denominator = sum(layer_packs)
            return (
                sum(float(value) * int(count) for value, count in zip(values, layer_packs))
                / denominator
                if denominator
                else 0.0
            )
        pa_values.extend(
            [
                total_blocks,
                total_packs,
                sum_metric(res_dict, f"{cache_kind}_pa_candidate_different_packs"),
                payload_beneficial,
                f"{payload_beneficial / total_packs:.8f}" if total_packs else "0.00000000",
                sum_metric(res_dict, f"{cache_kind}_pa_positive_delta_candidates"),
                sum_metric(res_dict, f"{cache_kind}_pa_nonpositive_delta_selected_packs"),
                sum_metric(res_dict, f"{cache_kind}_pa_budget_rejected_beneficial_packs"),
                selected_packs,
                f"{selected_packs / total_packs:.8f}" if total_packs else "0.00000000",
                f"{weighted_mean(f'{cache_kind}_pa_nearest_nmse_mean'):.10g}",
                f"{weighted_mean(f'{cache_kind}_pa_ceil_nmse_mean'):.10g}",
                f"{weighted_mean(f'{cache_kind}_pa_selected_nmse_mean'):.10g}",
                f"{nearest_sse:.10g}",
                f"{ceil_sse:.10g}",
                f"{selected_sse:.10g}",
                f"{error_budget_sse:.10g}",
                f"{used_delta_sse:.10g}",
                (
                    f"{max(0.0, used_delta_sse) / error_budget_sse:.8f}"
                    if error_budget_sse > 0
                    else "0.00000000"
                ),
                nearest_bits,
                ceil_bits,
                potential_bits,
                nearest_bits - potential_bits,
                selected_bits,
                nearest_bits - selected_bits,
                sum_metric(res_dict, f"{cache_kind}_pa_error_budget_violations"),
            ]
        )

    headers = metadata_headers + [
        "Num_Layers",
        "K_Original_Bytes",
        "V_Original_Bytes",
        "K_Quant_Payload_Bytes",
        "V_Quant_Payload_Bytes",
        "K_Quant_Total_Bytes",
        "V_Quant_Total_Bytes",
        "K_Quant_Global_CR",
        "V_Quant_Global_CR",
        "K_Bitpack_Payload_Before_Repack_Bytes",
        "V_Bitpack_Payload_Before_Repack_Bytes",
        "K_Encoded_Before_Repack_Bytes",
        "V_Encoded_Before_Repack_Bytes",
        "K_Encode_Before_Repack_Global_CR",
        "V_Encode_Before_Repack_Global_CR",
        "K_Compressed_Bytes",
        "V_Compressed_Bytes",
        "K_Global_CR",
        "V_Global_CR",
        "Overall_Global_CR",
        "K_Mem_Saved(%)",
        "V_Mem_Saved(%)",
        "K_Recent_FP_Bytes",
        "V_Recent_FP_Bytes",
        "K_Zero_Point_Bytes",
        "V_Zero_Point_Bytes",
        "K_Scale_Bytes",
        "V_Scale_Bytes",
        "K_Bitpack_Payload_Bytes",
        "V_Bitpack_Payload_Bytes",
        "K_Pack_Min_Bytes",
        "V_Pack_Min_Bytes",
        "K_Encode_Length_Bytes",
        "V_Encode_Length_Bytes",
        "K_Permutation_Metadata_Bytes",
        "V_Permutation_Metadata_Bytes",
        "K_Bucket_Metadata_Bytes",
        "V_Bucket_Metadata_Bytes",
        "K_Block_Metadata_Bytes",
        "V_Block_Metadata_Bytes",
        "K_Accounting_Error_Bytes",
        "V_Accounting_Error_Bytes",
        "K_Bitpack_Alignment_Bits",
        "V_Bitpack_Alignment_Bits",
        "K_Padding_Tokens",
        "V_Padding_Tokens",
        "K_Padding_Values",
        "V_Padding_Values",
        "K_Bit_Width_Hist_After",
        "V_Bit_Width_Hist_After",
        "K_Bit_Width_Hist_Before",
        "V_Bit_Width_Hist_Before",
        "Bucket_Occupancy_Hist",
        "Bucket_Empty_Rate",
        "K_PA_Total_Blocks", "K_PA_Total_Packs", "K_PA_Candidate_Different_Packs",
        "K_PA_Payload_Beneficial_Packs", "K_PA_Payload_Beneficial_Rate",
        "K_PA_Positive_Delta_Candidates", "K_PA_Nonpositive_Delta_Selected_Packs",
        "K_PA_Budget_Rejected_Beneficial_Packs",
        "K_PA_Ceil_Selected_Packs", "K_PA_Ceil_Selected_Rate",
        "K_PA_Nearest_NMSE", "K_PA_Ceil_NMSE", "K_PA_Selected_NMSE",
        "K_PA_Nearest_SSE", "K_PA_Ceil_SSE", "K_PA_Selected_SSE",
        "K_PA_Error_Budget_SSE", "K_PA_Used_Delta_SSE",
        "K_PA_Error_Budget_Utilization",
        "K_PA_Nearest_Payload_Bits", "K_PA_Ceil_Payload_Bits",
        "K_PA_Payload_Benefit_Ceiling_Bits", "K_PA_Max_Payload_Bits_Savable",
        "K_PA_Selected_Payload_Bits", "K_PA_Payload_Bits_Saved",
        "K_PA_Error_Budget_Violations",
        "V_PA_Total_Blocks", "V_PA_Total_Packs", "V_PA_Candidate_Different_Packs",
        "V_PA_Payload_Beneficial_Packs", "V_PA_Payload_Beneficial_Rate",
        "V_PA_Positive_Delta_Candidates", "V_PA_Nonpositive_Delta_Selected_Packs",
        "V_PA_Budget_Rejected_Beneficial_Packs",
        "V_PA_Ceil_Selected_Packs", "V_PA_Ceil_Selected_Rate",
        "V_PA_Nearest_NMSE", "V_PA_Ceil_NMSE", "V_PA_Selected_NMSE",
        "V_PA_Nearest_SSE", "V_PA_Ceil_SSE", "V_PA_Selected_SSE",
        "V_PA_Error_Budget_SSE", "V_PA_Used_Delta_SSE",
        "V_PA_Error_Budget_Utilization",
        "V_PA_Nearest_Payload_Bits", "V_PA_Ceil_Payload_Bits",
        "V_PA_Payload_Benefit_Ceiling_Bits", "V_PA_Max_Payload_Bits_Savable",
        "V_PA_Selected_Payload_Bits", "V_PA_Payload_Bits_Saved",
        "V_PA_Error_Budget_Violations",
        "K_Layer_CR_Min",
        "K_Layer_CR_Mean",
        "K_Layer_CR_Max",
        "K_Layer_CR_Std",
        "V_Layer_CR_Min",
        "V_Layer_CR_Mean",
        "V_Layer_CR_Max",
        "V_Layer_CR_Std",
        "Detailed_Report_Path",
        "Unified_Layer_Detail_Path",
    ]

    row_data = metadata_values + [
        len(res_dict.get("k_original_size", [])),
        k_original_bytes,
        v_original_bytes,
        sum_metric(res_dict, "k_quant_payload_size"),
        sum_metric(res_dict, "v_quant_payload_size"),
        k_quant_bytes,
        v_quant_bytes,
        f"{k_original_bytes / k_quant_bytes:.4f}" if k_quant_bytes else "",
        f"{v_original_bytes / v_quant_bytes:.4f}" if v_quant_bytes else "",
        sum_metric(res_dict, "k_bitpack_payload_size_before_repack"),
        sum_metric(res_dict, "v_bitpack_payload_size_before_repack"),
        k_before_repack_bytes,
        v_before_repack_bytes,
        (
            f"{k_original_bytes / k_before_repack_bytes:.4f}"
            if k_before_repack_bytes
            else ""
        ),
        (
            f"{v_original_bytes / v_before_repack_bytes:.4f}"
            if v_before_repack_bytes
            else ""
        ),
        k_compressed_bytes,
        v_compressed_bytes,
        f"{k_global_cr:.4f}",
        f"{v_global_cr:.4f}",
        f"{overall_global_cr:.4f}",
        f"{k_save_pct:.2f}%",
        f"{v_save_pct:.2f}%",
        components["k"]["recent_high_precision_size"],
        components["v"]["recent_high_precision_size"],
        components["k"]["quant_zero_point_size"],
        components["v"]["quant_zero_point_size"],
        components["k"]["quant_scale_size"],
        components["v"]["quant_scale_size"],
        components["k"]["bitpack_payload_size_after_repack"],
        components["v"]["bitpack_payload_size_after_repack"],
        components["k"]["bitpack_min_size_after_repack"],
        components["v"]["bitpack_min_size_after_repack"],
        components["k"]["bitpack_encode_len_size_after_repack"],
        components["v"]["bitpack_encode_len_size_after_repack"],
        components["k"]["permutation_metadata_size"],
        components["v"]["permutation_metadata_size"],
        components["k"]["bucket_metadata_size"],
        components["v"]["bucket_metadata_size"],
        components["k"]["block_metadata_size"],
        components["v"]["block_metadata_size"],
        k_compressed_bytes - accounting_totals["k"],
        v_compressed_bytes - accounting_totals["v"],
        sum_metric(res_dict, "k_bitpack_alignment_bits_after_repack"),
        sum_metric(res_dict, "v_bitpack_alignment_bits_after_repack"),
        sum_metric(res_dict, "k_bitpack_padding_tokens_after_repack"),
        sum_metric(res_dict, "v_bitpack_padding_tokens_after_repack"),
        sum_metric(res_dict, "k_bitpack_padding_values_after_repack"),
        sum_metric(res_dict, "v_bitpack_padding_values_after_repack"),
        json.dumps(k_width_hist, sort_keys=True),
        json.dumps(v_width_hist, sort_keys=True),
        json.dumps(k_width_hist_before, sort_keys=True),
        json.dumps(v_width_hist_before, sort_keys=True),
        json.dumps(bucket_occupancy_hist, sort_keys=True),
        f"{bucket_empty_rate:.8f}" if bucket_empty_rate != "" else "",
        *pa_values,
        *[format_float_or_blank(value) for value in k_layer_stats],
        *[format_float_or_blank(value) for value in v_layer_stats],
        f"{csv_path}",
        f"{layer_detail_path}",
    ]

    try:
        if file_exists:
            with open(summary_file, mode="r", newline="", encoding="utf-8") as f:
                existing_headers = next(csv.reader(f), None)
            if existing_headers != headers:
                raise ValueError(
                    "Global_Macro_Summary_v9.csv 表头与当前 schema 不一致"
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

    metadata_headers, metadata_values = result_metadata(
        args, round_idx, generated_at
    )
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


def append_to_layer_detail_csv(args, res_dict, round_idx):
    """将所有实验的逐层结果追加到统一明细表,便于一次性上传分析."""
    save_dir = "./csv_results"
    os.makedirs(save_dir, exist_ok=True)
    detail_path = os.path.join(save_dir, "Layer_Detail_v4.csv")
    file_exists = os.path.isfile(detail_path)

    num_layers = len(res_dict.get("k_original_size", []))
    if num_layers == 0:
        return None

    generated_at = datetime.datetime.now()
    metadata_headers, metadata_values = result_metadata(
        args, round_idx, generated_at
    )
    list_keys = [
        key
        for key, values in res_dict.items()
        if isinstance(values, list) and len(values) == num_layers
    ]
    headers = metadata_headers + ["Layer_Index"] + list_keys

    try:
        if file_exists:
            with open(detail_path, mode="r", newline="", encoding="utf-8") as f:
                existing_headers = next(csv.reader(f), None)
            if existing_headers != headers:
                raise ValueError(
                    "Layer_Detail_v4.csv 表头与当前 schema 不一致"
                )

        with open(detail_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            for layer_idx in range(num_layers):
                writer.writerow(
                    metadata_values
                    + [layer_idx]
                    + [res_dict[key][layer_idx] for key in list_keys]
                )
        return detail_path
    except Exception as e:
        print(f"❌ 写入统一逐层明细失败: {e}")
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
        "--k_error_budget", type=float, default=0.1,
        help="packing-aware 每层 K selected SSE 相对 nearest SSE 的增幅预算",
    )
    parser.add_argument(
        "--v_error_budget", type=float, default=0.1,
        help="packing-aware 每层 V selected SSE 相对 nearest SSE 的增幅预算",
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
    parser.add_argument(
        "--suite_id",
        type=str,
        default="manual",
        help="实验套件标识,例如 RQ1_full 或 RQ1_sanity",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="唯一运行标识;未指定时自动生成",
    )
    parser.add_argument(
        "--sample_id",
        type=int,
        default=None,
        help="输入缓存样本标识;未指定时使用从0开始的 round 索引",
    )

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
    if args.collect_round <= 0:
        parser.error("--collect_round must be a positive integer")
    if args.k_error_budget < 0 or args.v_error_budget < 0:
        parser.error("--k_error_budget and --v_error_budget must be non-negative")
    args.suite_id = safe_filename_component(args.suite_id) or "manual"
    if args.run_id is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")
        args.run_id = f"{args.suite_id}-{timestamp}-{uuid.uuid4().hex[:8]}"

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
    if args.scale_method == ScaleMethod.PO2_PACK_AWARE.value and (
        quant_method_enum != QuantMethod.PackKV
        or repack_method_enum != RepackMethod.BUCKET
        or args.bucket_score_method != BucketScoreMethod.K_SUM.value
    ):
        parser.error(
            "po2_pack_aware currently requires --quant_method PackKV "
            "--repack_method BUCKET --bucket_score_method k_sum"
        )

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
        k_error_budget=args.k_error_budget,
        v_error_budget=args.v_error_budget,
    )
    args.enable_save = True
    logger.info("=" * 50)
    logger.info(f"   开始压缩率 (CR) 评测: {args.model_name}")
    logger.info(f"   Context Length: {args.ctx_len}")
    logger.info(f"   K Scale: {args.k_scale}, V Scale: {args.v_scale}")
    logger.info(f"   Scale Method: {args.scale_method}")
    logger.info(
        f"   Packing-aware Layer SSE Budget: K={args.k_error_budget}, V={args.v_error_budget}"
    )
    logger.info(f"   High Precision Zero Point: {args.high_precision_zero_point}")
    logger.info(f"   Block Size: {args.block_size}, Pack Size: {args.pack_size}")
    logger.info(f"   Bucket Count: {args.bucket_count}")
    logger.info(f"   Bucket Score Method: {args.bucket_score_method}")
    logger.info(f"   Suite ID: {args.suite_id}")
    logger.info(f"   Run ID: {args.run_id}")
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

                    layer_detail_path = append_to_layer_detail_csv(
                        args, res, i + 1
                    )
                    if layer_detail_path:
                        print(
                            f"   统一逐层明细已追加至 : {layer_detail_path}"
                        )

                    summary_path = append_to_macro_summary_csv(
                        args,
                        res,
                        i + 1,
                        csv_path,
                        layer_detail_path,
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
