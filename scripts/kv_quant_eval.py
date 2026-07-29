"""在已保存的 KV Cache 上比较连续 scale 与 2^k scale 的量化误差.

该脚本只读取 evaluation.evaluation.save_extract_cache() 生成的 .pt 文件。
它不加载模型，也不把伪量化位宽当作 PackKV 的实际压缩率。
"""

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# 本脚本在下方自行按 K/V 和策略统计,关闭 compute.py 的混合全局统计.
os.environ["PACKKV_GLOBAL_K_STATS"] = "0"

from utils.compute import (  # noqa: E402
    QuantMethod,
    ScaleMethod,
    dequantize_ints,
    quant_ints,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="比较已保存 KV Cache 上不同 scale 策略的量化误差"
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--ctx_len", type=int, required=True)
    parser.add_argument("--cache_root", default="./dumped_cache")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument(
        "--quant_method",
        choices=[method.name for method in QuantMethod],
        default=QuantMethod.PackKV.name,
    )
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--k_scale", type=float, default=0.01)
    parser.add_argument("--v_scale", type=float, default=0.01)
    parser.add_argument(
        "--scale_methods",
        nargs="+",
        choices=[method.value for method in ScaleMethod],
        default=[method.value for method in ScaleMethod],
    )
    parser.add_argument(
        "--high_precision_zero_point",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="默认使用整数 zero point;开启后保存浮点 minimum",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="可选;指定时保存逐层结果,不指定则不写文件",
    )
    return parser.parse_args()


def numeric_pt_files(directory):
    files = {}
    for path in directory.glob("*.pt"):
        try:
            layer = int(path.stem)
        except ValueError as error:
            raise ValueError(f"非标准层文件名: {path.name}") from error
        files[layer] = path
    return files


def cache_paths(args):
    base = (
        Path(args.cache_root)
        / args.model_name
        / str(args.ctx_len)
        / str(args.round)
    )
    k_files = numeric_pt_files(base / "k")
    v_files = numeric_pt_files(base / "v")
    if not k_files or not v_files:
        raise FileNotFoundError(f"未找到完整缓存目录: {base}/[k|v]/*.pt")
    if k_files.keys() != v_files.keys():
        raise ValueError(
            f"K/V 层号不一致: K={sorted(k_files)}, V={sorted(v_files)}"
        )
    return [(layer, k_files[layer], v_files[layer]) for layer in sorted(k_files)]


def prepare_tensor(path, block_size):
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
        raise ValueError(f"{path} 应为四维 Tensor [B,H,L,D]")
    used_tokens = tensor.shape[2] - tensor.shape[2] % block_size
    if used_tokens == 0:
        raise ValueError(
            f"{path} 的 token 数 {tensor.shape[2]} 小于 block_size={block_size}"
        )
    return tensor[:, :, :used_tokens, :], tensor.shape[2] - used_tokens


def global_code_bits(quant_int):
    q_min = int(quant_int.min().item())
    q_max = int(quant_int.max().item())
    return max(1, math.ceil(math.log2(q_max - q_min + 1)))


def error_metrics(original, reconstructed):
    source = original.to(torch.float32)
    restored = reconstructed.to(torch.float32)
    delta = restored - source
    source_norm = torch.linalg.vector_norm(source)
    restored_norm = torch.linalg.vector_norm(restored)
    eps = torch.finfo(torch.float32).eps
    return {
        "mae": delta.abs().mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "max_abs": delta.abs().max().item(),
        "relative_l2": (
            torch.linalg.vector_norm(delta) / source_norm.clamp_min(eps)
        ).item(),
        "cosine": (
            (source.flatten() @ restored.flatten())
            / (source_norm * restored_norm).clamp_min(eps)
        ).item(),
    }


def evaluate_tensor(
    tensor,
    cache_kind,
    layer,
    quant_mode,
    scale_rel,
    scale_methods,
    block_size,
    high_precision_zero_point,
    truncated_tokens,
    k_distributions,
):
    _, _, continuous_scale = quant_ints(
        tensor,
        block_size,
        scale_rel,
        quant_mode,
        high_precision_zero_point,
        ScaleMethod.CONTINUOUS,
    )
    rows = []
    for scale_method in scale_methods:
        quant_int, quant_zero, quant_scale = quant_ints(
            tensor,
            block_size,
            scale_rel,
            quant_mode,
            high_precision_zero_point,
            scale_method,
        )
        reconstructed = dequantize_ints(
            quant_int,
            quant_zero,
            quant_scale,
            high_precision_zero_point,
        ).reshape_as(tensor)
        ratio = quant_scale.to(torch.float32) / continuous_scale.to(torch.float32)
        row = {
            "layer": layer,
            "cache": cache_kind,
            "scale_method": scale_method.value,
            "high_precision_zero_point": high_precision_zero_point,
            "used_tokens": tensor.shape[2],
            "truncated_tokens": truncated_tokens,
            "global_code_bits": global_code_bits(quant_int),
            "q_min": int(quant_int.min().item()),
            "q_max": int(quant_int.max().item()),
            "scale_ratio_mean": ratio.mean().item(),
            "scale_ratio_min": ratio.min().item(),
            "scale_ratio_max": ratio.max().item(),
            "scale_ratio_lt_1_fraction": (ratio < 1).float().mean().item(),
            "scale_ratio_gt_1_fraction": (ratio > 1).float().mean().item(),
        }
        row.update(error_metrics(tensor, reconstructed))
        if scale_method == ScaleMethod.CONTINUOUS:
            row["k_min"] = ""
            row["k_max"] = ""
        else:
            exponents = torch.log2(quant_scale.to(torch.float32))
            row["k_min"] = int(exponents.min().item())
            row["k_max"] = int(exponents.max().item())
            unique_ks, counts = torch.unique(
                exponents.to(torch.int64).flatten(), return_counts=True
            )
            k_distributions[(cache_kind, scale_method.value)].update(
                {
                    int(exponent): int(count)
                    for exponent, count in zip(
                        unique_ks.tolist(), counts.tolist()
                    )
                }
            )
        rows.append(row)
    return rows


def print_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["cache"], row["scale_method"])].append(row)
    print(
        "cache  scale_method     MAE(mean)    RMSE(mean)   rel_L2(mean) "
        "cosine(mean) bits(max)"
    )
    for (cache_kind, scale_method), items in sorted(grouped.items()):
        mean = lambda key: sum(item[key] for item in items) / len(items)
        print(
            f"{cache_kind:5s}  {scale_method:15s} "
            f"{mean('mae'):11.6g} {mean('rmse'):12.6g} "
            f"{mean('relative_l2'):12.6g} {mean('cosine'):12.8f} "
            f"{max(item['global_code_bits'] for item in items):9d}"
        )


def print_k_distributions(k_distributions):
    print("\n按 cache 和 scale_method 分组的 2^k 分布:")
    if not k_distributions:
        print("未选择 2^k scale 策略.")
        return
    for (cache_kind, scale_method), counter in sorted(k_distributions.items()):
        total = sum(counter.values())
        print(f"\n[{cache_kind} / {scale_method}] scale 参数数: {total}")
        for exponent in sorted(counter):
            count = counter[exponent]
            percentage = count / total * 100
            print(
                f"  k={exponent:4d} (scale=2^{exponent:<3d}): "
                f"{count:10d},占比 {percentage:6.2f}%"
            )


def write_csv(rows, output_csv):
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"逐层结果已保存: {path}")


def main():
    args = parse_args()
    if args.block_size <= 0:
        raise ValueError("block_size 必须大于 0")
    for name, value in (("k_scale", args.k_scale), ("v_scale", args.v_scale)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须是有限正数")

    quant_method = QuantMethod[args.quant_method]
    scale_methods = [ScaleMethod(value) for value in args.scale_methods]
    rows = []
    k_distributions = defaultdict(Counter)
    for layer, k_path, v_path in cache_paths(args):
        k_tensor, k_truncated = prepare_tensor(k_path, args.block_size)
        v_tensor, v_truncated = prepare_tensor(v_path, args.block_size)
        if k_tensor.shape[0] != v_tensor.shape[0] or k_tensor.shape[2] != v_tensor.shape[2]:
            raise ValueError(
                f"第 {layer} 层 K/V 的 batch 或 token 维不一致: "
                f"K={tuple(k_tensor.shape)}, V={tuple(v_tensor.shape)}"
            )
        rows.extend(
            evaluate_tensor(
                k_tensor,
                "K",
                layer,
                quant_method.value[0],
                args.k_scale,
                scale_methods,
                args.block_size,
                args.high_precision_zero_point,
                k_truncated,
                k_distributions,
            )
        )
        rows.extend(
            evaluate_tensor(
                v_tensor,
                "V",
                layer,
                quant_method.value[1],
                args.v_scale,
                scale_methods,
                args.block_size,
                args.high_precision_zero_point,
                v_truncated,
                k_distributions,
            )
        )

    print_summary(rows)
    print_k_distributions(k_distributions)
    if args.output_csv:
        write_csv(rows, args.output_csv)


if __name__ == "__main__":
    main()
