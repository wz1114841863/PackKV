import torch
import atexit
import math
import os
from typing import Tuple, Optional, List, Dict
from enum import Enum
from collections import defaultdict
from dataclasses import dataclass

GLOBAL_K_COUNTER = defaultdict(int)


@atexit.register
def print_k_distribution():
    """在脚本运行结束时,自动打印 2^k 的分布情况"""
    if os.environ.get("PACKKV_GLOBAL_K_STATS", "1") == "0":
        return
    print("\n" + "=" * 50)
    print("🎯 [量化硬件分析] 实际使用的 2^k 分布统计:")
    total = sum(GLOBAL_K_COUNTER.values())
    if total == 0:
        print("未记录到量化数据...")
        return

    # 按 k 的大小排序打印
    for k in sorted(GLOBAL_K_COUNTER.keys()):
        count = GLOBAL_K_COUNTER[k]
        percentage = (count / total) * 100
        print(
            f"  k = {k:4d} (Scale = 2^{k:<4d}) : {count:10d} 个 Block, 占比 {percentage:5.2f}%"
        )
    print("=" * 50 + "\n")


class QuantMode(Enum):
    """定义不同的量化维度, 决定了量化时沿着那个维度求极值"""

    # LayerQuant = "LayerQuant"
    BlockQuant = "BlockQuant"
    ChannelQuant = "ChannelQuant"
    TokenQuant = "TokenQuant"
    VectorQuant = "VectorQuant"


class QuantMethod(Enum):
    """指明了具体的量化策略"""

    KIVI = (QuantMode.ChannelQuant, QuantMode.TokenQuant)
    PackKV = (QuantMode.TokenQuant, QuantMode.TokenQuant)


class ScaleMethod(Enum):
    """量化步长策略."""

    CONTINUOUS = "continuous"
    PO2_NEAREST = "po2_nearest"
    PO2_FLOOR = "po2_floor"
    PO2_CEIL = "po2_ceil"
    PO2_PACK_AWARE = "po2_pack_aware"


class QuantMetadataFormat(Enum):
    """量化参数的物理存储格式."""

    NATIVE = "native"
    PO2_COMPACT = "po2_compact"


class BucketScoreMethod(Enum):
    """硬件 Bucket 的整数特征提取方式."""

    COMBINED_SUM = "combined_sum"
    K_SUM = "k_sum"
    V_SUM = "v_sum"
    KV_2D = "kv_2d"


class RepackMethod(Enum):
    """重排策略"""

    GREEDY = "Greedy"
    MEDIAN = "Median"
    NONE = "None"
    # 添加新的方法
    BUCKET = "Bucket"


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """将RoPE注入到Query和Key"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_single(
    t: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids=None,
    unsqueeze_dim=1,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    t_embed = (t * cos) + (rotate_half(t) * sin)
    return t_embed


def safe_cat(t1, t2, dim):
    if t1 is None and t2 is None:
        return None
    if t1 is None:
        return t2.clone()
    if t2 is None:
        return t1.clone()
    return torch.cat([t1, t2], dim=dim)


def cut_tensor(
    buffer, new_tensor, block_size, recent_size, dim=2
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """将动态增长的 Cache 拼接 (safe_cat) 起来,并按照 block_size 进行切分.
    只有凑够了一个完整的 Block(且排除了 recent_size 即最近的无需压缩的高精度 Token),才会被送入后续的量化流程.
    这模拟了硬件执行流中,数据从片上 SRAM (Buffer) 满载后,被压缩写回大容量 DRAM 的过程."""
    buffer = safe_cat(buffer, new_tensor, dim)
    len_ = buffer.shape[dim]
    if block_size <= 0 or recent_size < 0:
        raise ValueError("block_size must be positive and recent_size non-negative")
    # 仅压缩在保留 recent_size 后仍完整存在的 block；与离线 CR 的
    # get_compressible_prefix_length 口径一致。
    to_compress_block_num = max(0, (len_ - recent_size) // block_size)
    to_compress = None
    if to_compress_block_num > 0:
        to_compress = buffer[:, :, : to_compress_block_num * block_size, :]
        buffer = buffer[:, :, to_compress_block_num * block_size :, :]
    return to_compress, buffer


def cut_tensor_ctx_len_0(
    buffer, new_tensor, block_size, recent_size, dim=2
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    buffer = safe_cat(buffer, new_tensor, dim)
    len_ = buffer.shape[dim]
    res_num = len_ % block_size
    to_compress_block_num = (len_ + block_size - res_num - recent_size) // block_size
    to_compress = None
    if to_compress_block_num > 0:
        to_compress = buffer[: to_compress_block_num * block_size, :, :, :]
        buffer = buffer[to_compress_block_num * block_size :, :, :, :]
    return to_compress, buffer


def calculate_aware_quant_scale(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    quant_scale_rel: float,
    po2_strategy: str = "precision",
) -> torch.Tensor:
    """
    计算结合了容器感知和硬件二次幂对齐的量化 Scale
    po2_strategy:
        - "precision": (保精度) 允许偶尔扩展 1 bit 的空间来降低量化误差
        - "memory": (保内存) 强制向上取整,绝对不突破原定配置的 bit 位数
        - "none": 不使用二次幂限制 (原版逻辑)
    """
    tensor_range = max_val - min_val
    eps = 1e-7

    # 原始的软件期望 Scale
    raw_scale = torch.clamp(tensor_range * quant_scale_rel, min=eps)

    if po2_strategy == "none":
        return raw_scale

    # 第一步:探底,计算初始配置期望使用的位宽 (Target Bits)
    max_int_init = tensor_range / raw_scale
    target_bits = torch.ceil(torch.log2(max_int_init + 1.0))
    target_bits = torch.clamp(target_bits, min=1.0)

    # 第二步:容器拉伸,计算刚好填满该位宽容器的理想 Scale
    c_max = torch.exp2(target_bits) - 1.0
    ideal_scale = tensor_range / torch.clamp(c_max, min=1.0)
    ideal_scale = torch.clamp(ideal_scale, min=eps)

    # 第三步:二次幂逼近,根据传入的策略做出抉择
    if po2_strategy == "memory":
        # 向上取整:Scale 变大,切分变粗,一定能装进 Target Bits,但可能有浪费
        k = torch.ceil(torch.log2(ideal_scale))
    elif po2_strategy == "precision":
        # 四舍五入:优先找最近的二次幂.如果变小,切分变细,动态打包时会自动扩充 1 bit
        k = torch.round(torch.log2(ideal_scale))
    else:
        raise ValueError(f"Unknown po2_strategy: {po2_strategy}")

    # 返回硬件友好的 2^k 作为最终的量化步长
    return torch.exp2(k)


def quant_ints(
    tensor: torch.Tensor,
    block_size: int,
    quant_scale_rel: float,  # Relative Quantization Scale(相对量化比例)
    quant_mode: QuantMode,
    high_precision_zero_point: bool = False,
    scale_method: ScaleMethod = ScaleMethod.CONTINUOUS,
    record_k_stats: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tensor.ndim != 4:
        raise ValueError("tensor must have shape [batch, head, sequence, dim]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not math.isfinite(quant_scale_rel) or quant_scale_rel <= 0:
        raise ValueError("quant_scale_rel must be a finite positive number")
    if not tensor.is_floating_point():
        raise TypeError("tensor must have a floating-point dtype")
    if not torch.isfinite(tensor).all():
        raise ValueError("tensor contains NaN or Inf")
    if tensor.shape[2] % block_size:
        raise ValueError("sequence length must be divisible by block_size")
    # 根据block_size进行reshape
    tensor = tensor.reshape(
        tensor.shape[0], tensor.shape[1], -1, block_size, tensor.shape[3]
    )  # [1, 32, 128, 128] -> [1, 32, 8, 16, 128]
    quant_dim = QUANT_DIM[quant_mode.value]

    min_val = tensor
    max_val = tensor
    for i in quant_dim:
        min_val = min_val.min(dim=i, keepdim=True).values
        max_val = max_val.max(dim=i, keepdim=True).values

    scale_raw = torch.clamp((max_val - min_val) * quant_scale_rel, min=1e-5)
    if scale_method == ScaleMethod.CONTINUOUS:
        quant_scale = scale_raw
    else:
        # log2/指数选择使用 FP32,最终 scale 转回原 metadata dtype.
        log2_scale = torch.log2(scale_raw.to(torch.float32))
        if scale_method == ScaleMethod.PO2_NEAREST:
            k = torch.round(log2_scale)
        elif scale_method == ScaleMethod.PO2_FLOOR:
            k = torch.floor(log2_scale)
        elif scale_method == ScaleMethod.PO2_CEIL:
            k = torch.ceil(log2_scale)
        elif scale_method == ScaleMethod.PO2_PACK_AWARE:
            raise ValueError(
                "po2_pack_aware requires joint K/V quantization; "
                "use packing_aware_quantize_kv in the offline CR path"
            )
        else:
            raise ValueError(f"Unknown scale_method: {scale_method}")
        dtype_info = torch.finfo(scale_raw.dtype)
        min_k = math.floor(math.log2(1e-5))
        max_k = math.floor(math.log2(dtype_info.max))
        k = torch.clamp(k, min=min_k, max=max_k)
        quant_scale = torch.exp2(k).to(scale_raw.dtype)
        if record_k_stats and os.environ.get("PACKKV_GLOBAL_K_STATS", "1") != "0":
            with torch.no_grad():
                unique_ks, counts = torch.unique(k.flatten(), return_counts=True)
                for exponent, count in zip(unique_ks.tolist(), counts.tolist()):
                    GLOBAL_K_COUNTER[int(exponent)] += count

    if high_precision_zero_point:
        # 返回浮点 minimum;对应反量化公式 q * scale + minimum.
        value_quant = ((tensor - min_val) / quant_scale).round()
    else:
        # 返回整数 zero point;对应反量化公式 (q + zero) * scale.
        min_int = (min_val / quant_scale).round()
        value_quant = (tensor / quant_scale).round() - min_int
        min_val = min_int

    if not (
        torch.isfinite(value_quant).all()
        and torch.isfinite(min_val).all()
        and torch.isfinite(quant_scale).all()
    ):
        zero_point_hint = (
            " Try high_precision_zero_point=True for constant or very narrow "
            "low-precision blocks."
            if not high_precision_zero_point
            else ""
        )
        raise FloatingPointError(
            "quantization produced NaN or Inf; the selected metadata dtype "
            "cannot represent this scale/zero point." + zero_point_hint
        )

    return value_quant, min_val, quant_scale


def dequantize_ints(
    quant_int: torch.Tensor,
    quant_zero: torch.Tensor,
    quant_scale: torch.Tensor,
    high_precision_zero_point: bool = False,
) -> torch.Tensor:
    """与 quant_ints 的两种 zero-point 语义严格对应."""
    if high_precision_zero_point:
        result = quant_int * quant_scale + quant_zero
    else:
        result = (quant_int + quant_zero) * quant_scale
    if not torch.isfinite(result).all():
        raise FloatingPointError("dequantization produced NaN or Inf")
    return result


def quant_ints_2k(
    tensor: torch.Tensor,
    block_size: int,
    quant_scale_rel: float,
    quant_mode,
    high_precision_zero_point: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """兼容旧调用方的 nearest-2^k 量化入口."""
    return quant_ints(
        tensor=tensor,
        block_size=block_size,
        quant_scale_rel=quant_scale_rel,
        quant_mode=quant_mode,
        high_precision_zero_point=high_precision_zero_point,
        scale_method=ScaleMethod.PO2_NEAREST,
    )


def quant_ints_2k_error(
    tensor: torch.Tensor,
    block_size: int,
    quant_scale_rel: float,
    quant_mode: QuantMode,
    high_precision_zero_point: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    硬件友好的 2^k 移位量化器
    保持与原版 quant_ints 完全一致的输入参数和返回值结构.

    感知逻辑存在问题, 运行KIVI时报错存在NAN值.
    """
    assert (
        tensor.shape[2] % block_size == 0
    ), "Tensor shape is not divisible by block size"

    # 1. 按照 block_size 进行 reshape
    # [B, H, SeqLen, D] -> [B, H, SeqLen//block_size, block_size, D]
    tensor_reshaped = tensor.reshape(
        tensor.shape[0], tensor.shape[1], -1, block_size, tensor.shape[3]
    )

    # 获取需要求极值的维度
    quant_dims = QUANT_DIM[quant_mode.value]

    # 2. 提取局部极值
    min_ = tensor_reshaped
    max_ = tensor_reshaped
    for dim in quant_dims:
        min_ = min_.min(dim=dim, keepdim=True).values
        max_ = max_.max(dim=dim, keepdim=True).values

    # ==========================================
    # 核心注入: 容器感知 + 2^k 移位量化
    # ==========================================
    # 加入 1e-5 防止 Padding 死头导致的除零崩溃
    range_ = torch.clamp(max_ - min_, min=1e-5)

    # (1) 探底:计算初始期望的位宽
    scale_init = range_ * quant_scale_rel
    max_int_init = range_ / scale_init  # 把range_抵消了.
    target_bits = torch.clamp(torch.ceil(torch.log2(max_int_init + 1)), min=1.0)

    # (2) 容器拉伸:计算该位宽下能把容器撑满的理想 Scale
    c_max = (2**target_bits) - 1
    scale_ideal = range_ / c_max

    # (3) 二次幂逼近 (保精度 Round 策略)
    k = torch.round(torch.log2(scale_ideal))
    quant_scale = torch.pow(2.0, k)  # 最终 Scale = 2^k

    # 在 GPU 上异步计数,不拖慢推理
    with torch.no_grad():
        # 在 GPU 上找出当前这批数据有哪几种 k,以及各自的数量
        unique_ks, counts = torch.unique(k.flatten(), return_counts=True)
        # 转移回 CPU 并更新全局字典 (非常快,因为 unique_ks 通常长度不超过 10)
        for kv, c in zip(unique_ks.tolist(), counts.tolist()):
            GLOBAL_K_COUNTER[int(kv)] += c

    # 3. 执行真正的量化 (除以 quant_scale 等价于硬件层的右移)
    min_ints = (min_ / quant_scale).round_()
    q_ints = (tensor_reshaped / quant_scale).round_()

    # 返回: (相对量化整数, 零点整数, 比例尺)
    return q_ints - min_ints, min_ints, quant_scale


def quant_ints_throughput(
    tensor: torch.Tensor,
    block_size: int,
    quant_scale_rel: float,
    quant_mode: QuantMode,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert (
        tensor.shape[2] % block_size == 0
    ), "Tensor shape is not divisible by block size"
    tensor = tensor.reshape(
        tensor.shape[0], tensor.shape[1], -1, block_size, tensor.shape[3]
    )
    quant_dim = QUANT_DIM[quant_mode.value]

    min_val = torch.amin(tensor, dim=quant_dim, keepdim=True)
    max_val = torch.amax(tensor, dim=quant_dim, keepdim=True)

    quant_scale = (max_val - min_val) * quant_scale_rel
    min_int = (min_val / quant_scale).round()
    value_quant = (tensor / quant_scale).round() - min_int
    min_val = min_int

    return value_quant, min_val, quant_scale


def quant(
    tensor: torch.Tensor, quant_dims: List[int], quant_scale_rel: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    min_ = tensor
    max_ = tensor
    for dim in quant_dims:
        min_ = min_.min(dim=dim, keepdim=True).values
        max_ = max_.max(dim=dim, keepdim=True).values
    quant_scale = (max_ - min_) * quant_scale_rel
    min_ints = (min_ / quant_scale).round_()  # .to(torch.int8)
    quant_ints = (tensor / quant_scale).round_()  # .to(torch.int8)
    return quant_ints - min_ints, min_ints, quant_scale


def quant_error(
    error_cache: torch.Tensor,
    buffer: torch.Tensor,
    new_tensor: torch.Tensor,
    block_size: int,
    recent_size: int,
    quant_scale_rel: float,
    quant_mode: QuantMode,
    high_precision_zero_point: bool = False,
    scale_method: ScaleMethod = ScaleMethod.CONTINUOUS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """伪量化, 计算量化过程带来的误差, 用于算法维度的验证和补偿"""
    to_compress, in_buffer = cut_tensor(
        buffer, new_tensor, block_size, recent_size, dim=2
    )

    if to_compress is not None:
        quant_int, quant_zero, quant_scale = quant_ints(
            to_compress,
            block_size,
            quant_scale_rel,
            quant_mode,
            high_precision_zero_point,
            scale_method,
        )
        to_compress = dequantize_ints(
            quant_int,
            quant_zero,
            quant_scale,
            high_precision_zero_point,
        )
        to_compress = to_compress.reshape(
            to_compress.shape[0], to_compress.shape[1], -1, to_compress.shape[4]
        )

    return safe_cat(error_cache, to_compress, dim=2), in_buffer


def quant_error_kv_bucket_repacked(
    k_error_cache: Optional[torch.Tensor],
    v_error_cache: Optional[torch.Tensor],
    k_buffer: Optional[torch.Tensor],
    v_buffer: Optional[torch.Tensor],
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    block_size: int,
    recent_size: int,
    k_quant_scale_rel: float,
    v_quant_scale_rel: float,
    k_quant_mode: QuantMode,
    v_quant_mode: QuantMode,
    high_precision_zero_point: bool = False,
    scale_method: ScaleMethod = ScaleMethod.CONTINUOUS,
    bucket_count: int = 4,
    bucket_score_method: BucketScoreMethod = BucketScoreMethod.K_SUM,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """在线精度评测的量化后 BUCKET 重排参考路径.

    新压缩 block 的 K/V 量化整数生成同一 bucket 置换，反量化后的 K/V
    使用该置换共同重排并保存。prefill 当前调用仍返回原 token 顺序，避免
    破坏 causal mask；后续单 token decode 返回重排后的缓存。这样实际执行
    decode 置换不变性，而不是仅把 ``repack_method`` 记录在配置中。

    当前参考路径不支持在已有重排缓存上继续多 token/chunked prefill，因为
    仅凭 bucket counts 不能恢复原始时间顺序。
    """
    if new_k.shape != new_v.shape:
        raise ValueError("new K/V shapes must match for shared bucket repacking")
    if scale_method == ScaleMethod.PO2_PACK_AWARE:
        raise ValueError("packing-aware quantization uses its dedicated joint path")

    is_prefill = new_k.shape[2] != 1
    if is_prefill and (k_error_cache is not None or v_error_cache is not None):
        raise ValueError(
            "chunked prefill after bucket-repacked cache is not supported"
        )

    k_to_compress, k_in_buffer = cut_tensor(
        k_buffer, new_k, block_size, recent_size, dim=2
    )
    v_to_compress, v_in_buffer = cut_tensor(
        v_buffer, new_v, block_size, recent_size, dim=2
    )
    if (k_to_compress is None) != (v_to_compress is None):
        raise ValueError("K/V compression boundaries diverged")
    if k_in_buffer.shape[2] != v_in_buffer.shape[2]:
        raise ValueError("K/V buffer lengths diverged")

    k_attention_cache = k_error_cache
    v_attention_cache = v_error_cache
    if k_to_compress is not None:
        if k_to_compress.shape != v_to_compress.shape:
            raise ValueError("K/V compressible shapes diverged")
        k_quant_int, k_quant_zero, k_quant_scale = quant_ints(
            k_to_compress,
            block_size,
            k_quant_scale_rel,
            k_quant_mode,
            high_precision_zero_point,
            scale_method,
        )
        v_quant_int, v_quant_zero, v_quant_scale = quant_ints(
            v_to_compress,
            block_size,
            v_quant_scale_rel,
            v_quant_mode,
            high_precision_zero_point,
            scale_method,
        )
        quant_blocks = torch.cat(
            [
                _quant_tensor_to_blocks(k_quant_int),
                _quant_tensor_to_blocks(v_quant_int),
            ],
            dim=2,
        )
        permutation, _ = _build_bucket_permutation(
            quant_blocks,
            bucket_count,
            bucket_score_method,
        )

        k_reconstructed = dequantize_ints(
            k_quant_int,
            k_quant_zero,
            k_quant_scale,
            high_precision_zero_point,
        )
        v_reconstructed = dequantize_ints(
            v_quant_int,
            v_quant_zero,
            v_quant_scale,
            high_precision_zero_point,
        )

        def apply_permutation(tensor: torch.Tensor) -> torch.Tensor:
            indices = permutation.view(
                1, 1, permutation.shape[0], permutation.shape[1], 1
            ).expand_as(tensor)
            return torch.gather(tensor, 3, indices)

        k_repacked = apply_permutation(k_reconstructed).reshape_as(k_to_compress)
        v_repacked = apply_permutation(v_reconstructed).reshape_as(v_to_compress)
        k_reconstructed = k_reconstructed.reshape_as(k_to_compress)
        v_reconstructed = v_reconstructed.reshape_as(v_to_compress)

        # 存储始终采用重排顺序；只有当前 prefill Attention 使用原顺序。
        k_error_cache = safe_cat(k_error_cache, k_repacked, dim=2)
        v_error_cache = safe_cat(v_error_cache, v_repacked, dim=2)
        if is_prefill:
            k_attention_cache = k_reconstructed
            v_attention_cache = v_reconstructed
        else:
            k_attention_cache = k_error_cache
            v_attention_cache = v_error_cache

    return (
        k_error_cache,
        v_error_cache,
        k_in_buffer,
        v_in_buffer,
        k_attention_cache,
        v_attention_cache,
    )


def quant_error_kv_packing_aware(
    k_error_cache: Optional[torch.Tensor],
    v_error_cache: Optional[torch.Tensor],
    k_buffer: Optional[torch.Tensor],
    v_buffer: Optional[torch.Tensor],
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    block_size: int,
    recent_size: int,
    pack_size: int,
    k_quant_scale_rel: float,
    v_quant_scale_rel: float,
    k_quant_mode: QuantMode,
    v_quant_mode: QuantMode,
    high_precision_zero_point: bool = False,
    k_error_budget: float = 0.1,
    v_error_budget: float = 0.1,
    bucket_count: int = 4,
    bucket_score_method: BucketScoreMethod = BucketScoreMethod.K_SUM,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """在线精度评测使用的 K/V 联合伪量化入口.

    Bucket 置换只用于估计 pack cost 和选择 nearest/ceil。反量化结果保持
    原 token 顺序返回给 Attention，因此不会把 decode 的置换不变性错误
    推广到带 causal mask 的 prefill。流式 decode 后续形成的新 block 各自
    使用一次预算；这与离线整层 oracle 的预算作用域并不完全相同。
    """
    if new_k.shape[:3] != new_v.shape[:3]:
        raise ValueError("new K/V batch, head and sequence dimensions must match")
    k_to_compress, k_in_buffer = cut_tensor(
        k_buffer, new_k, block_size, recent_size, dim=2
    )
    v_to_compress, v_in_buffer = cut_tensor(
        v_buffer, new_v, block_size, recent_size, dim=2
    )
    if (k_to_compress is None) != (v_to_compress is None):
        raise ValueError("K/V compression boundaries diverged")
    if k_in_buffer.shape[2] != v_in_buffer.shape[2]:
        raise ValueError("K/V buffer lengths diverged")

    if k_to_compress is not None:
        if k_to_compress.shape[2] != v_to_compress.shape[2]:
            raise ValueError("K/V compressible lengths diverged")
        (
            k_quant_int,
            k_quant_zero,
            k_quant_scale,
            v_quant_int,
            v_quant_zero,
            v_quant_scale,
            _,
            _,
            _,
            _,
        ) = packing_aware_quantize_kv(
            k_to_compress,
            v_to_compress,
            block_size,
            pack_size,
            k_quant_scale_rel,
            v_quant_scale_rel,
            k_quant_mode,
            v_quant_mode,
            high_precision_zero_point,
            k_error_budget,
            v_error_budget,
            bucket_count,
            bucket_score_method,
        )
        k_reconstructed = dequantize_ints(
            k_quant_int,
            k_quant_zero,
            k_quant_scale,
            high_precision_zero_point,
        ).reshape(
            k_to_compress.shape[0],
            k_to_compress.shape[1],
            -1,
            k_to_compress.shape[3],
        )
        v_reconstructed = dequantize_ints(
            v_quant_int,
            v_quant_zero,
            v_quant_scale,
            high_precision_zero_point,
        ).reshape(
            v_to_compress.shape[0],
            v_to_compress.shape[1],
            -1,
            v_to_compress.shape[3],
        )
        k_error_cache = safe_cat(k_error_cache, k_reconstructed, dim=2)
        v_error_cache = safe_cat(v_error_cache, v_reconstructed, dim=2)

    return k_error_cache, v_error_cache, k_in_buffer, v_in_buffer


def quant_without_repacking(
    error_cache: torch.Tensor,
    buffer: torch.Tensor,
    new_tensor: torch.Tensor,
    block_size: int,
    recent_size: int,
    quant_scale_rel: float,
    quant_mode: QuantMode,
    high_precision_zero_point: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    退守重建版:剥离一切切块与重排逻辑,仅进行最基础的有损量化测试.
    """
    # ==========================================
    # 1. 极简拼接:先把所有的 Token 拼在一起
    # ==========================================
    if buffer is not None:
        full_tensor = torch.cat([buffer, new_tensor], dim=2)
    else:
        full_tensor = new_tensor

    # ==========================================
    # 2. 隔离保护区:切出需要量化的部分和受保护的最新部分
    # ==========================================
    seq_len = full_tensor.shape[2]

    if seq_len <= recent_size:
        # 如果长度还没超过保护区,什么都不做,直接返回全精度
        return safe_cat(error_cache, full_tensor, dim=2), None

    # 把超过保护区的数据切下来去量化
    to_compress = full_tensor[:, :, :-recent_size, :]
    # 把最近的 Token 留作未来的 buffer
    in_buffer = full_tensor[:, :, -recent_size:, :]

    # ==========================================
    # 3. 最原始的 8-bit / 4-bit 均匀量化 (Absmax Quantization)
    # ==========================================
    # 我们设定为 8-bit 量化,最大整数为 127
    bits = 8
    max_int = 2 ** (bits - 1) - 1  # 127

    # 沿着 Token 维度 (dim=2) 和 Channel 维度 (dim=3) 找到绝对最大值
    # 这里我们采用极其安全的 Per-Head 量化,避免跨 Head 污染
    abs_max = to_compress.abs().amax(dim=(2), keepdim=True)

    # 防除零保护
    abs_max = torch.clamp(abs_max, min=1e-5)

    # 计算 Scale
    scale = abs_max / max_int

    # 量化 (取整)
    quantized = torch.clamp((to_compress / scale).round(), -max_int, max_int)

    # 反量化 (解压)
    dequantized = quantized * scale

    # ==========================================
    # 4. 完美拼接返回
    # ==========================================
    # 将量化后的旧历史与之前的超长历史拼接
    final_cache = safe_cat(error_cache, dequantized, dim=2)

    return final_cache, in_buffer


def print_quant_setting(logger):
    logger.info(QUANT_DIM)


def _batched_pick(
    tensor: torch.Tensor, indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Picks a vector from each batch element and returns the picked vectors and the remaining tensors.
    tensor: (B, N, D)
    indices: (B,)
    Returns: (picked_vectors (B, D), remaining_tensor (B, N-1, D))
    """
    B, N, D = tensor.shape
    device = tensor.device

    # 通过索引扩展, 摘取目标向量, 得到picked_vectors
    idx_expanded = indices.view(B, 1, 1).expand(-1, 1, D)
    picked_vectors = torch.gather(tensor, 1, idx_expanded).squeeze(1)

    if N == 1:
        # 如果池子里只剩1个向量, 直接返回抽出的向量和一个空的剩余张量
        remained_tensor = torch.empty((B, 0, D), dtype=tensor.dtype, device=device)
        return picked_vectors, remained_tensor

    # 利用布尔掩码, 生成剩余张量
    mask = torch.ones(B, N, device=device, dtype=torch.bool)
    batch_indices = torch.arange(B, device=device)
    mask[batch_indices, indices] = False
    remained_tensor = tensor[mask].view(B, N - 1, D)

    return picked_vectors, remained_tensor


def greedy_repacking(blocks: torch.Tensor, pack_len: int) -> torch.Tensor:
    """基于余弦相似度的贪心聚类.
    它不断计算候选向量与当前均值向量的相似度, 将最相似的打包在一起.
    这种方式压缩率极高, 但涉及大量的浮点矩阵乘法和排序, 计算复杂度高."""
    B, N, D = blocks.shape
    remaining_blocks = blocks.clone().to(torch.float32)
    pack_num = N // pack_len
    repacked_packs_list = []

    for _ in range(pack_num):
        # 先求出当前剩余所有向量的均值
        mean_vectors = remaining_blocks.mean(dim=1, keepdim=True).round()
        # 计算所有候选向量与均值的余弦相似度, 挑出最接近均值的向量
        cosine_sim = torch.nn.functional.cosine_similarity(
            remaining_blocks, mean_vectors, dim=2, eps=1e-8
        )
        max_sim_indices = torch.argmax(cosine_sim, dim=1)
        seed_vectors, remaining_blocks = _batched_pick(
            remaining_blocks, max_sim_indices
        )
        # 基于最小位宽增量进行贪心扩充
        pack_tensor_list = [seed_vectors.unsqueeze(1)]
        mins_ = seed_vectors
        maxs_ = seed_vectors
        for _ in range(pack_len - 1):
            if remaining_blocks.shape[1] == 0:
                break
            current_mins = mins_.unsqueeze(1)
            current_maxs = maxs_.unsqueeze(1)
            pre_status_num = torch.ceil(torch.log2(current_maxs - current_mins + 1))

            all_possible_max = torch.max(remaining_blocks, current_maxs)
            all_possible_min = torch.min(remaining_blocks, current_mins)
            #
            all_possible_bit_num = torch.ceil(
                torch.log2(all_possible_max - all_possible_min + 1)
            )
            all_possible_bit_num_increase = (all_possible_bit_num - pre_status_num).sum(
                dim=2
            )
            # 选出让位宽增加最少的那一个向量
            selected_vector_indices = torch.argmin(all_possible_bit_num_increase, dim=1)

            selected_vectors, remaining_blocks = _batched_pick(
                remaining_blocks, selected_vector_indices
            )

            pack_tensor_list.append(selected_vectors.unsqueeze(1))

            mins_ = torch.min(mins_, selected_vectors)
            maxs_ = torch.max(maxs_, selected_vectors)

        pack = torch.cat(pack_tensor_list, dim=1)
        repacked_packs_list.append(pack)

    repacked_blocks = torch.cat(repacked_packs_list, dim=1)
    return repacked_blocks.to(torch.int32)


def median_repacking(blocks: torch.Tensor) -> torch.Tensor:
    """取每个Token向量中后半部分V Cache的中位数, 然后按中位数大小进行降序重排.
    相比于贪心算法,寻找中位数并排序的过程更加硬件友好, 可以用更少的逻辑门和排序网络
    来实现，在吞吐量上优势明显."""
    B, N, D = blocks.shape
    half_vec_len = D // 2
    # 切片提取V Cache
    # 在注意力机制中, V Cache的数值分布特征通常比K Cache对最终输出的影响更直接，
    # 且分布有其特有的规律
    v_part = blocks[:, :, half_vec_len:]
    # 对V Cache的特征维度求中位数 median_values
    median_values = torch.median(v_part, dim=2).values
    # 根据中位数对V Cache降序排序
    _, sorted_indices = torch.sort(median_values, dim=1, descending=True)
    sorted_indices_expanded = sorted_indices.unsqueeze(2).expand(B, N, D)
    # 拿到排序索引后, 对原始 blocks 进行一次性重排
    repacked_blocks = torch.gather(blocks, 1, sorted_indices_expanded)
    return repacked_blocks


@dataclass(frozen=True)
class RepackMetadataStats:
    """重排参考格式需要随编码流保存的元数据."""

    bucket_count: int = 0
    bucket_count_field_bits: int = 0
    bucket_metadata_bits: int = 0
    bucket_counts: Tuple[Tuple[int, ...], ...] = ()
    bucket_occupancy_histogram: Optional[Dict[int, int]] = None
    bucket_score_method: str = ""
    k_subbucket_count: int = 0
    v_subbucket_count: int = 0

    @property
    def bucket_metadata_bytes(self) -> int:
        if not self.bucket_counts:
            return 0
        # 每个基础 block 的 header 独立字节对齐,便于硬件随机访问.
        bits_per_block = (
            (self.bucket_count - 1) * self.bucket_count_field_bits
        )
        return len(self.bucket_counts) * ((bits_per_block + 7) // 8)


def _validate_bucket_count(num_buckets: int, tokens_per_block: int) -> None:
    if num_buckets < 2:
        raise ValueError("num_buckets must be at least 2")
    if num_buckets > tokens_per_block:
        raise ValueError("num_buckets must not exceed tokens per block")
    if num_buckets & (num_buckets - 1):
        raise ValueError("num_buckets must be a power of two")


def _equal_width_bucket_ids(
    scores: torch.Tensor,
    num_buckets: int,
) -> torch.Tensor:
    """用整数等宽阈值生成 bucket ID;桶数为1时返回全0."""
    if num_buckets == 1:
        return torch.zeros_like(scores, dtype=torch.int64)
    score_min = scores.min(dim=1, keepdim=True).values
    score_max = scores.max(dim=1, keepdim=True).values
    span = score_max - score_min + 1
    boundary_ids = torch.arange(
        1,
        num_buckets,
        device=scores.device,
        dtype=torch.int64,
    )
    shift = int(math.log2(num_buckets))
    offsets = (
        span.unsqueeze(2) * boundary_ids.view(1, 1, -1)
        + num_buckets
        - 1
    ) >> shift
    thresholds = score_min.unsqueeze(2) + offsets
    return (scores.unsqueeze(2) >= thresholds).sum(dim=2).to(torch.int64)


def _build_bucket_permutation(
    blocks: torch.Tensor,
    num_buckets: int,
    score_method: BucketScoreMethod,
) -> Tuple[torch.Tensor, RepackMetadataStats]:
    """生成稳定 FIFO Bucket 的 token 置换及其可解码元数据."""
    if blocks.ndim != 3:
        raise ValueError("blocks must have shape [block, token, feature]")
    block_count, tokens_per_block, feature_dim = blocks.shape
    if block_count <= 0 or tokens_per_block <= 0 or feature_dim <= 0:
        raise ValueError("blocks dimensions must be non-zero")
    if feature_dim % 2 != 0:
        raise ValueError("K/V concatenated feature dimension must be even")
    if blocks.is_floating_point() and not torch.isfinite(blocks).all():
        raise ValueError("blocks contains NaN or Inf")
    _validate_bucket_count(num_buckets, tokens_per_block)
    if isinstance(score_method, str):
        score_method = BucketScoreMethod(score_method)
    if score_method == BucketScoreMethod.KV_2D and num_buckets < 4:
        raise ValueError("kv_2d requires at least 4 buckets")

    integer_blocks = blocks.to(torch.int64)
    half_feature_dim = feature_dim // 2
    k_scores = integer_blocks[:, :, :half_feature_dim].sum(dim=2)
    v_scores = integer_blocks[:, :, half_feature_dim:].sum(dim=2)
    k_subbucket_count = 0
    v_subbucket_count = 0
    if score_method == BucketScoreMethod.COMBINED_SUM:
        bucket_ids = _equal_width_bucket_ids(k_scores + v_scores, num_buckets)
    elif score_method == BucketScoreMethod.K_SUM:
        bucket_ids = _equal_width_bucket_ids(k_scores, num_buckets)
    elif score_method == BucketScoreMethod.V_SUM:
        bucket_ids = _equal_width_bucket_ids(v_scores, num_buckets)
    elif score_method == BucketScoreMethod.KV_2D:
        total_bucket_bits = int(math.log2(num_buckets))
        k_bucket_bits = (total_bucket_bits + 1) // 2
        v_bucket_bits = total_bucket_bits // 2
        k_subbucket_count = 1 << k_bucket_bits
        v_subbucket_count = 1 << v_bucket_bits
        k_bucket_ids = _equal_width_bucket_ids(k_scores, k_subbucket_count)
        v_bucket_ids = _equal_width_bucket_ids(v_scores, v_subbucket_count)
        bucket_ids = k_bucket_ids * v_subbucket_count + v_bucket_ids
    else:
        raise ValueError(f"Unknown bucket score method: {score_method}")

    # argsort 对 bucket ID 稳定排序，等价于按桶号顺序读取稳定 FIFO。
    permutation = torch.argsort(bucket_ids, dim=1, stable=True)
    bucket_counts_tensor = torch.stack(
        [(bucket_ids == bucket_idx).sum(dim=1) for bucket_idx in range(num_buckets)],
        dim=1,
    )
    count_field_bits = max(1, math.ceil(math.log2(tokens_per_block + 1)))
    metadata_bits = block_count * (num_buckets - 1) * count_field_bits
    counts_cpu = bucket_counts_tensor.cpu().tolist()
    bucket_counts = tuple(tuple(int(value) for value in row) for row in counts_cpu)
    occupancy_histogram: Dict[int, int] = defaultdict(int)
    for row in bucket_counts:
        for count in row:
            occupancy_histogram[count] += 1
    metadata = RepackMetadataStats(
        bucket_count=num_buckets,
        bucket_count_field_bits=count_field_bits,
        bucket_metadata_bits=metadata_bits,
        bucket_counts=bucket_counts,
        bucket_occupancy_histogram=dict(sorted(occupancy_histogram.items())),
        bucket_score_method=score_method.value,
        k_subbucket_count=k_subbucket_count,
        v_subbucket_count=v_subbucket_count,
    )
    return permutation, metadata


def _apply_token_permutation(
    blocks: torch.Tensor, permutation: torch.Tensor
) -> torch.Tensor:
    if permutation.shape != blocks.shape[:2]:
        raise ValueError("permutation shape must match [block, token]")
    indices = permutation.unsqueeze(2).expand(-1, -1, blocks.shape[2])
    return torch.gather(blocks, 1, indices)


def apply_quant_metadata_token_permutation(
    metadata: torch.Tensor, permutation: torch.Tensor
) -> torch.Tensor:
    """Apply a block-local token permutation to quantization metadata.

    ``quant_ints`` returns five-dimensional metadata tensors.  TokenQuant
    metadata has shape ``[B, 1, block, token, 1]`` and must follow the same
    token order as the encoded q values.  Metadata from a non-token quantizer
    has a singleton token dimension and is invariant to token repacking.
    """
    if metadata.ndim != 5:
        raise ValueError("quantization metadata must be a 5-D tensor")
    if permutation.ndim != 2:
        raise ValueError("permutation must have shape [block, token]")
    if metadata.shape[2] != permutation.shape[0]:
        raise ValueError("metadata block count does not match permutation")
    if metadata.shape[3] == 1:
        return metadata
    if metadata.shape[3] != permutation.shape[1]:
        raise ValueError("metadata token count does not match permutation")
    indices = permutation.view(
        1, 1, permutation.shape[0], permutation.shape[1], 1
    ).expand_as(metadata)
    return torch.gather(metadata, 3, indices)


def build_kv_bucket_permutation(
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    bucket_count: int = 4,
    bucket_score_method: BucketScoreMethod = BucketScoreMethod.K_SUM,
) -> Tuple[torch.Tensor, RepackMetadataStats]:
    """Build the single bucket permutation shared by K/V and their metadata."""
    if k_tensor.shape != v_tensor.shape or k_tensor.ndim != 5:
        raise ValueError("quantized K/V must have matching 5-D shapes")
    blocks = torch.cat(
        [_quant_tensor_to_blocks(k_tensor), _quant_tensor_to_blocks(v_tensor)],
        dim=2,
    )
    return _build_bucket_permutation(
        blocks, bucket_count, BucketScoreMethod(bucket_score_method)
    )


def bucket_repacking(
    blocks: torch.Tensor,
    num_buckets: int = 4,
    score_method: BucketScoreMethod = BucketScoreMethod.COMBINED_SUM,
    return_metadata: bool = False,
):
    """两遍式/稳定 FIFO 的硬件 Bucket 重排参考模型.

    输入 ``blocks`` 为 [block, token, K+V feature].K/V 已沿最后一维拼接,
    因而这里只生成一次 bucket ID,天然保证 K/V 使用相同置换.

    硬件映射:
      1. 按 score_method 生成 combined/K/V 整数和;KV_2D 分别保留K/V score;
      2. 每个基础 block 求 score 的 min/max;
      3. 桶数限制为 2^n,等宽阈值仅需常数乘法/加法和右移;
      4. 比较器阵列生成 bucket ID,稳定写入多路 FIFO;
      5. 按固定 bucket ID 顺序读出,不执行全排序.

    最后一个桶计数可由 block token 总数和其他桶计数推导,因此元数据只
    保存 ``num_buckets - 1`` 个计数.动态阈值只在编码端用于路由,不需要
    随压缩流保存.
    """
    permutation, metadata = _build_bucket_permutation(
        blocks, num_buckets, score_method
    )
    repacked_blocks = _apply_token_permutation(blocks, permutation)
    if return_metadata:
        return repacked_blocks, metadata
    return repacked_blocks


def encode_bucket_metadata(metadata: RepackMetadataStats) -> bytes:
    """序列化每个基础 block 的前 ``bucket_count - 1`` 个桶计数.

    每个 block 独立补齐到字节边界，与 ``bucket_metadata_bytes`` 的压缩率
    统计口径一致；最后一个桶计数由 block token 总数减去其余计数得到。
    """
    if not metadata.bucket_counts:
        return b""
    output = bytearray()
    for counts in metadata.bucket_counts:
        if len(counts) != metadata.bucket_count:
            raise ValueError("bucket count row length mismatch")
        if any(int(count) < 0 for count in counts):
            raise ValueError("bucket counts must be non-negative")
        stored_counts = [int(count) for count in counts[:-1]]
        output.extend(
            _pack_unsigned_fields(
                stored_counts,
                [metadata.bucket_count_field_bits] * len(stored_counts),
            )
        )
    encoded = bytes(output)
    if len(encoded) != metadata.bucket_metadata_bytes:
        raise AssertionError("bucket metadata byte accounting mismatch")
    return encoded


def decode_bucket_metadata(
    data: bytes,
    block_count: int,
    tokens_per_block: int,
    bucket_count: int,
    score_method: BucketScoreMethod = BucketScoreMethod.K_SUM,
) -> RepackMetadataStats:
    """解码 bucket count header，并验证边界、总数与 padding bits."""
    if block_count <= 0 or tokens_per_block <= 0:
        raise ValueError("block_count and tokens_per_block must be positive")
    _validate_bucket_count(bucket_count, tokens_per_block)
    if isinstance(score_method, str):
        score_method = BucketScoreMethod(score_method)
    count_field_bits = max(1, math.ceil(math.log2(tokens_per_block + 1)))
    stored_count_count = bucket_count - 1
    bytes_per_block = (stored_count_count * count_field_bits + 7) // 8
    expected_bytes = block_count * bytes_per_block
    if len(data) != expected_bytes:
        raise ValueError(
            f"bucket metadata length mismatch: expected {expected_bytes}, got {len(data)}"
        )

    bucket_counts = []
    occupancy_histogram: Dict[int, int] = defaultdict(int)
    for block_idx in range(block_count):
        begin = block_idx * bytes_per_block
        stored_counts = _unpack_unsigned_fields(
            data[begin : begin + bytes_per_block],
            [count_field_bits] * stored_count_count,
        )
        last_count = tokens_per_block - sum(stored_counts)
        if last_count < 0:
            raise ValueError("decoded bucket counts exceed tokens_per_block")
        row = tuple(stored_counts + [last_count])
        if any(count > tokens_per_block for count in row):
            raise ValueError("decoded bucket count exceeds tokens_per_block")
        bucket_counts.append(row)
        for count in row:
            occupancy_histogram[count] += 1

    total_bucket_bits = int(math.log2(bucket_count))
    k_subbucket_count = 0
    v_subbucket_count = 0
    if score_method == BucketScoreMethod.KV_2D:
        if bucket_count < 4:
            raise ValueError("kv_2d requires at least 4 buckets")
        k_subbucket_count = 1 << ((total_bucket_bits + 1) // 2)
        v_subbucket_count = 1 << (total_bucket_bits // 2)
    return RepackMetadataStats(
        bucket_count=bucket_count,
        bucket_count_field_bits=count_field_bits,
        bucket_metadata_bits=(
            block_count * stored_count_count * count_field_bits
        ),
        bucket_counts=tuple(bucket_counts),
        bucket_occupancy_histogram=dict(sorted(occupancy_histogram.items())),
        bucket_score_method=score_method.value,
        k_subbucket_count=k_subbucket_count,
        v_subbucket_count=v_subbucket_count,
    )


def hardware_bucket_repacking(
    blocks: torch.Tensor,
    num_main_buckets: int = 4,
    score_method: BucketScoreMethod = BucketScoreMethod.COMBINED_SUM,
    return_metadata: bool = False,
):
    """兼容旧入口;统一使用 ``bucket_repacking`` 硬件参考模型."""
    return bucket_repacking(
        blocks,
        num_buckets=num_main_buckets,
        score_method=score_method,
        return_metadata=return_metadata,
    )


@dataclass(frozen=True)
class BitPackStats:
    """一个连续 bitstream 编码后的分项存储开销."""

    payload_bits: int
    pack_min_bits: int
    encode_length_bits: int
    byte_alignment_bits: int
    total_bytes: int
    code_value_bits: int
    pack_count: int
    padded_token_count: int
    padded_value_count: int
    bit_width_histogram: Dict[int, int]

    @property
    def payload_bytes(self) -> int:
        return (self.payload_bits + 7) // 8

    @property
    def pack_min_bytes(self) -> int:
        return (self.pack_min_bits + 7) // 8

    @property
    def encode_length_bytes(self) -> int:
        return (self.encode_length_bits + 7) // 8


@dataclass(frozen=True)
class BitPackedCacheStream:
    """可实际解码的单个 K 或 V bit-packed 参考流.

    三个字节串分别对应 ``BitPackStats`` 已统计的 payload、pack minimum
    和 encode length。shape、pack_len 以及字段位宽属于模型/编码器配置描述符，
    不重复写入每层数据流；round-trip 解码不会借用原始 tensor。
    """

    payload: bytes
    pack_mins: bytes
    encode_lengths: bytes
    token_count: int
    feature_dim: int
    pack_len: int
    padded_token_count: int
    code_value_bits: int
    encode_length_field_bits: int
    signed_values: bool

    @property
    def total_bytes(self) -> int:
        return len(self.payload) + len(self.pack_mins) + len(self.encode_lengths)


@dataclass(frozen=True)
class CompactQuantMetadataStream:
    """整数 zero point 与 2^k exponent 的两个连续窄字段流.

    shape 和字段位宽属于模型/编码器描述符，不重复写入每层数据流。
    """

    zero_points: bytes
    exponents: bytes
    shape: Tuple[int, ...]
    zero_point_bits: int
    exponent_bits: int

    @property
    def count(self) -> int:
        return math.prod(self.shape)

    @property
    def total_bytes(self) -> int:
        return len(self.zero_points) + len(self.exponents)

    @property
    def alignment_bits(self) -> int:
        raw_bits = self.count * (self.zero_point_bits + self.exponent_bits)
        return self.total_bytes * 8 - raw_bits


def _pack_unsigned_fields(values: List[int], widths: List[int]) -> bytes:
    """按 LSB-first 连续 bitstream 编码非负整数."""
    if len(values) != len(widths):
        raise ValueError("values and widths must have the same length")
    accumulator = 0
    accumulator_bits = 0
    output = bytearray()
    for value, width in zip(values, widths):
        value = int(value)
        width = int(width)
        if width < 0:
            raise ValueError("field width must be non-negative")
        if value < 0 or (width == 0 and value != 0) or value >= (1 << width):
            raise ValueError(f"value {value} does not fit in {width} bits")
        accumulator |= value << accumulator_bits
        accumulator_bits += width
        while accumulator_bits >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            accumulator_bits -= 8
    if accumulator_bits:
        output.append(accumulator & 0xFF)
    return bytes(output)


def _unpack_unsigned_fields(data: bytes, widths: List[int]) -> List[int]:
    """解码 ``_pack_unsigned_fields`` 产生的 LSB-first bitstream."""
    expected_bits = sum(int(width) for width in widths)
    expected_bytes = (expected_bits + 7) // 8
    if len(data) != expected_bytes:
        raise ValueError(
            f"bitstream length mismatch: expected {expected_bytes}, got {len(data)}"
        )
    raw = int.from_bytes(data, byteorder="little", signed=False)
    values = []
    offset = 0
    for width in widths:
        width = int(width)
        if width < 0:
            raise ValueError("field width must be non-negative")
        mask = (1 << width) - 1 if width else 0
        values.append((raw >> offset) & mask)
        offset += width
    if raw >> expected_bits:
        raise ValueError("non-zero byte-alignment padding bits")
    return values


def _signed_to_unsigned(value: int, width: int) -> int:
    lower = -(1 << (width - 1))
    upper = (1 << (width - 1)) - 1
    if value < lower or value > upper:
        raise ValueError(f"signed value {value} does not fit in {width} bits")
    return value if value >= 0 else (1 << width) + value


def _unsigned_to_signed(value: int, width: int) -> int:
    sign_bit = 1 << (width - 1)
    return value - (1 << width) if value & sign_bit else value


def _integer_storage_bits(min_value: int, max_value: int) -> int:
    """返回覆盖给定整数范围所需的最小定长位宽."""
    if min_value > max_value:
        raise ValueError("min_value must not exceed max_value")
    if min_value >= 0:
        return max(1, math.ceil(math.log2(max_value + 1)))

    bits = 1
    while min_value < -(1 << (bits - 1)) or max_value > (1 << (bits - 1)) - 1:
        bits += 1
    return bits


def encode_compact_quant_metadata(
    quant_zero: torch.Tensor,
    quant_scale: torch.Tensor,
    zero_point_bits: int,
    exponent_bits: int,
) -> CompactQuantMetadataStream:
    """把整数 zero point 和严格 2^k scale 编码为真实字节流."""
    if quant_zero.shape != quant_scale.shape:
        raise ValueError("quant_zero and quant_scale must have matching shapes")
    if zero_point_bits <= 0 or exponent_bits <= 0:
        raise ValueError("metadata field widths must be positive")
    if quant_zero.numel() == 0:
        raise ValueError("quantization metadata must not be empty")
    if quant_zero.is_floating_point():
        if not torch.isfinite(quant_zero).all() or not torch.equal(
            quant_zero, quant_zero.round()
        ):
            raise ValueError("compact metadata requires integer zero points")
    if not quant_scale.is_floating_point():
        raise TypeError("quant_scale must be floating point")
    if not torch.isfinite(quant_scale).all() or (quant_scale <= 0).any():
        raise ValueError("quant_scale must contain finite positive values")

    scale_fp32 = quant_scale.detach().to(torch.float32)
    exponents = torch.round(torch.log2(scale_fp32))
    if not torch.equal(scale_fp32, torch.exp2(exponents)):
        raise ValueError("compact metadata requires exact power-of-two scales")

    zero_values = [
        _signed_to_unsigned(int(value), zero_point_bits)
        for value in quant_zero.detach().to(torch.int64).cpu().flatten().tolist()
    ]
    exponent_values = [
        _signed_to_unsigned(int(value), exponent_bits)
        for value in exponents.to(torch.int64).cpu().flatten().tolist()
    ]
    return CompactQuantMetadataStream(
        zero_points=_pack_unsigned_fields(
            zero_values, [zero_point_bits] * len(zero_values)
        ),
        exponents=_pack_unsigned_fields(
            exponent_values, [exponent_bits] * len(exponent_values)
        ),
        shape=tuple(quant_zero.shape),
        zero_point_bits=zero_point_bits,
        exponent_bits=exponent_bits,
    )


def decode_compact_quant_metadata(
    stream: CompactQuantMetadataStream,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """无原始 tensor 参与地恢复整数 zero point 与 2^k scale."""
    if stream.count <= 0:
        raise ValueError("invalid compact metadata shape")
    zero_unsigned = _unpack_unsigned_fields(
        stream.zero_points, [stream.zero_point_bits] * stream.count
    )
    exponent_unsigned = _unpack_unsigned_fields(
        stream.exponents, [stream.exponent_bits] * stream.count
    )
    zero = torch.tensor(
        [_unsigned_to_signed(value, stream.zero_point_bits) for value in zero_unsigned],
        dtype=torch.int64,
    ).reshape(stream.shape)
    exponent = torch.tensor(
        [_unsigned_to_signed(value, stream.exponent_bits) for value in exponent_unsigned],
        dtype=torch.int64,
    ).reshape(stream.shape)
    return zero, torch.exp2(exponent.to(torch.float32))


def verify_compact_quant_metadata_roundtrip(
    quant_zero: torch.Tensor,
    quant_scale: torch.Tensor,
    zero_point_bits: int,
    exponent_bits: int,
) -> CompactQuantMetadataStream:
    """编码并逐值核对窄字段 metadata，返回真实字节流用于计费."""
    stream = encode_compact_quant_metadata(
        quant_zero, quant_scale, zero_point_bits, exponent_bits
    )
    decoded_zero, decoded_scale = decode_compact_quant_metadata(stream)
    expected_zero = quant_zero.detach().to(device="cpu", dtype=torch.int64)
    expected_scale = quant_scale.detach().to(device="cpu", dtype=torch.float32)
    if not torch.equal(decoded_zero, expected_zero):
        raise AssertionError("zero-point metadata round-trip mismatch")
    if not torch.equal(decoded_scale, expected_scale):
        raise AssertionError("scale exponent metadata round-trip mismatch")
    return stream


@dataclass(frozen=True)
class IntegerFieldProfile:
    """硬件字段的整数范围、尾部分位数和精确直方图."""

    count: int
    min_value: Optional[int]
    p0001: Optional[int]
    p001: Optional[int]
    p999: Optional[int]
    p9999: Optional[int]
    max_value: Optional[int]
    required_bits: int
    histogram: Dict[int, int]


@dataclass(frozen=True)
class HardwareQuantizationProfile:
    """单个 K/V Cache 的硬件量化字段画像."""

    quantized: IntegerFieldProfile
    zero_point: IntegerFieldProfile
    exponent: IntegerFieldProfile
    non_integer_zero_point_count: int
    non_po2_scale_count: int


def _histogram_percentile(histogram: Dict[int, int], quantile: float) -> Optional[int]:
    """返回离散直方图的 nearest-rank 分位数."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    total = sum(int(count) for count in histogram.values())
    if total <= 0:
        return None
    rank = max(1, math.ceil(quantile * total))
    cumulative = 0
    for value, count in sorted(histogram.items()):
        cumulative += int(count)
        if cumulative >= rank:
            return int(value)
    raise AssertionError("histogram percentile rank was not reached")


def profile_integer_histogram(histogram: Dict[int, int]) -> IntegerFieldProfile:
    """把整数直方图转换为可直接用于字段位宽决策的画像."""
    normalized = {
        int(value): int(count)
        for value, count in histogram.items()
        if int(count) > 0
    }
    count = sum(normalized.values())
    if count == 0:
        return IntegerFieldProfile(
            count=0,
            min_value=None,
            p0001=None,
            p001=None,
            p999=None,
            p9999=None,
            max_value=None,
            required_bits=0,
            histogram={},
        )
    min_value = min(normalized)
    max_value = max(normalized)
    return IntegerFieldProfile(
        count=count,
        min_value=min_value,
        p0001=_histogram_percentile(normalized, 0.0001),
        p001=_histogram_percentile(normalized, 0.001),
        p999=_histogram_percentile(normalized, 0.999),
        p9999=_histogram_percentile(normalized, 0.9999),
        max_value=max_value,
        required_bits=_integer_storage_bits(min_value, max_value),
        histogram=dict(sorted(normalized.items())),
    )


def _integer_tensor_histogram(tensor: torch.Tensor, field_name: str) -> Dict[int, int]:
    if tensor.numel() == 0:
        return {}
    if tensor.is_floating_point():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{field_name} contains NaN or Inf")
        if not torch.equal(tensor, tensor.round()):
            raise ValueError(f"{field_name} must contain integer-valued data")
    values = tensor.detach().to(torch.int64)
    unique_values, counts = torch.unique(values, return_counts=True)
    return {
        int(value): int(count)
        for value, count in zip(unique_values.cpu().tolist(), counts.cpu().tolist())
    }


def profile_hardware_quantization(
    quant_int: torch.Tensor,
    quant_zero: torch.Tensor,
    quant_scale: torch.Tensor,
) -> HardwareQuantizationProfile:
    """统计 q、整数 zero point 和可精确表示的 2^k exponent.

    continuous scale 不会被强行解释成 exponent；所有非严格 2^k 的 scale
    单独计入 ``non_po2_scale_count``，避免为硬件字段位宽制造假数据。
    """
    if quant_scale.numel() == 0:
        exponent_histogram = {}
        non_po2_scale_count = 0
    else:
        if not quant_scale.is_floating_point():
            raise TypeError("quant_scale must be floating point")
        if not torch.isfinite(quant_scale).all() or (quant_scale <= 0).any():
            raise ValueError("quant_scale must contain finite positive values")
        scale_fp32 = quant_scale.detach().to(torch.float32)
        rounded_exponents = torch.round(torch.log2(scale_fp32))
        exact_po2 = scale_fp32 == torch.exp2(rounded_exponents)
        non_po2_scale_count = int((~exact_po2).sum().item())
        exponent_histogram = _integer_tensor_histogram(
            rounded_exponents[exact_po2], "power-of-two exponent"
        )

    if quant_zero.numel() == 0:
        integer_zero_histogram = {}
        non_integer_zero_point_count = 0
    else:
        if quant_zero.is_floating_point():
            if not torch.isfinite(quant_zero).all():
                raise ValueError("zero point contains NaN or Inf")
            integer_zero_mask = quant_zero == quant_zero.round()
            non_integer_zero_point_count = int(
                (~integer_zero_mask).sum().item()
            )
            integer_zero_histogram = _integer_tensor_histogram(
                quant_zero[integer_zero_mask], "integer zero point"
            )
        else:
            non_integer_zero_point_count = 0
            integer_zero_histogram = _integer_tensor_histogram(
                quant_zero, "integer zero point"
            )

    return HardwareQuantizationProfile(
        quantized=profile_integer_histogram(
            _integer_tensor_histogram(quant_int, "quantized value")
        ),
        zero_point=profile_integer_histogram(integer_zero_histogram),
        exponent=profile_integer_histogram(exponent_histogram),
        non_integer_zero_point_count=non_integer_zero_point_count,
        non_po2_scale_count=non_po2_scale_count,
    )


def _single_cache_bit_pack_stats(
    values: torch.Tensor,
    pack_len: int,
    padded_token_count: int,
) -> BitPackStats:
    if pack_len <= 0:
        raise ValueError("pack_len must be positive")
    if values.ndim != 2 or values.shape[0] % pack_len != 0:
        raise ValueError("values must be [token, feature] and divisible by pack_len")

    packs = values.view(-1, pack_len, values.shape[1])
    pack_mins = packs.min(dim=1).values
    pack_maxs = packs.max(dim=1).values
    widths = torch.ceil(torch.log2(pack_maxs - pack_mins + 1)).to(torch.int64)

    payload_bits = int(widths.sum().item()) * pack_len
    global_min = int(values.min().item())
    global_max = int(values.max().item())
    code_value_bits = _integer_storage_bits(global_min, global_max)
    pack_min_bits = pack_mins.numel() * code_value_bits
    encode_length_field_bits = max(1, math.ceil(math.log2(code_value_bits + 1)))
    encode_length_bits = widths.numel() * encode_length_field_bits

    component_bytes = (
        (payload_bits + 7) // 8
        + (pack_min_bits + 7) // 8
        + (encode_length_bits + 7) // 8
    )
    raw_bits = payload_bits + pack_min_bits + encode_length_bits
    byte_alignment_bits = component_bytes * 8 - raw_bits
    unique_widths, counts = torch.unique(widths, return_counts=True)
    histogram = {
        int(width): int(count)
        for width, count in zip(unique_widths.tolist(), counts.tolist())
    }

    return BitPackStats(
        payload_bits=payload_bits,
        pack_min_bits=pack_min_bits,
        encode_length_bits=encode_length_bits,
        byte_alignment_bits=byte_alignment_bits,
        total_bytes=component_bytes,
        code_value_bits=code_value_bits,
        pack_count=packs.shape[0],
        padded_token_count=padded_token_count,
        padded_value_count=padded_token_count * values.shape[1],
        bit_width_histogram=histogram,
    )


def bit_pack_stats(
    blocks: torch.Tensor, pack_len: int
) -> Tuple[BitPackStats, BitPackStats]:
    """统计K/V payload及其解码元数据,不包含量化scale/zero-point."""
    if blocks.ndim != 3:
        raise ValueError("blocks must have shape [block, token, feature]")
    if blocks.shape[2] % 2 != 0:
        raise ValueError("K/V concatenated feature dimension must be even")

    half_vec_len = blocks.shape[2] // 2
    flattened = blocks.flatten(0, 1).to(torch.int64)
    padded_token_count = (-flattened.shape[0]) % pack_len
    if padded_token_count:
        # 重复最后一个向量不会扩大最后一个pack的数值范围.
        padding = flattened[-1:].expand(padded_token_count, -1)
        flattened = torch.cat([flattened, padding], dim=0)

    k_values = flattened[:, :half_vec_len]
    v_values = flattened[:, half_vec_len:]
    return (
        _single_cache_bit_pack_stats(k_values, pack_len, padded_token_count),
        _single_cache_bit_pack_stats(v_values, pack_len, padded_token_count),
    )


def _encode_single_cache_bitstream(
    values: torch.Tensor,
    pack_len: int,
    padded_token_count: int,
) -> BitPackedCacheStream:
    """真实编码一个 [token, feature] 整数量化 Cache."""
    if values.ndim != 2 or values.shape[0] % pack_len != 0:
        raise ValueError("values must be [token, feature] and divisible by pack_len")
    if values.is_floating_point():
        if not torch.isfinite(values).all():
            raise ValueError("values contains NaN or Inf")
        if not torch.equal(values, values.round()):
            raise ValueError("bit packing requires integer-valued input")
    integer_values = values.to(torch.int64).cpu()
    packs = integer_values.view(-1, pack_len, integer_values.shape[1])
    pack_mins = packs.min(dim=1).values
    pack_maxs = packs.max(dim=1).values
    widths = torch.ceil(torch.log2(pack_maxs - pack_mins + 1)).to(torch.int64)

    global_min = int(integer_values.min().item())
    global_max = int(integer_values.max().item())
    code_value_bits = _integer_storage_bits(global_min, global_max)
    encode_length_field_bits = max(
        1, math.ceil(math.log2(code_value_bits + 1))
    )

    signed_values = global_min < 0
    minimum_values = [
        (
            _signed_to_unsigned(int(value), code_value_bits)
            if signed_values
            else int(value)
        )
        for value in pack_mins.flatten().tolist()
    ]
    minimum_widths = [code_value_bits] * len(minimum_values)
    width_values = [int(value) for value in widths.flatten().tolist()]
    width_widths = [encode_length_field_bits] * len(width_values)

    payload_values: List[int] = []
    payload_widths: List[int] = []
    for pack_idx in range(packs.shape[0]):
        for feature_idx in range(packs.shape[2]):
            width = int(widths[pack_idx, feature_idx].item())
            minimum = int(pack_mins[pack_idx, feature_idx].item())
            for token_idx in range(pack_len):
                payload_values.append(
                    int(packs[pack_idx, token_idx, feature_idx].item()) - minimum
                )
                payload_widths.append(width)

    return BitPackedCacheStream(
        payload=_pack_unsigned_fields(payload_values, payload_widths),
        pack_mins=_pack_unsigned_fields(minimum_values, minimum_widths),
        encode_lengths=_pack_unsigned_fields(width_values, width_widths),
        token_count=values.shape[0] - padded_token_count,
        feature_dim=values.shape[1],
        pack_len=pack_len,
        padded_token_count=padded_token_count,
        code_value_bits=code_value_bits,
        encode_length_field_bits=encode_length_field_bits,
        signed_values=signed_values,
    )


def bit_pack_encode(
    blocks: torch.Tensor, pack_len: int
) -> Tuple[BitPackedCacheStream, BitPackedCacheStream]:
    """把 [block, token, K+V feature] 真正编码为 K/V 字节流.

    padding 与 ``bit_pack_stats`` 相同：连续展平基础 block 后，仅在流末尾
    重复最后一个 token，使 token 数可被 pack_len 整除。
    """
    if pack_len <= 0:
        raise ValueError("pack_len must be positive")
    if blocks.ndim != 3 or blocks.shape[0] <= 0 or blocks.shape[1] <= 0:
        raise ValueError("blocks must have non-empty [block, token, feature] shape")
    if blocks.shape[2] <= 0 or blocks.shape[2] % 2 != 0:
        raise ValueError("K/V concatenated feature dimension must be positive and even")

    half_vec_len = blocks.shape[2] // 2
    flattened = blocks.flatten(0, 1)
    padded_token_count = (-flattened.shape[0]) % pack_len
    if padded_token_count:
        flattened = torch.cat(
            [flattened, flattened[-1:].expand(padded_token_count, -1)], dim=0
        )
    return (
        _encode_single_cache_bitstream(
            flattened[:, :half_vec_len], pack_len, padded_token_count
        ),
        _encode_single_cache_bitstream(
            flattened[:, half_vec_len:], pack_len, padded_token_count
        ),
    )


def bit_pack_decode(stream: BitPackedCacheStream) -> torch.Tensor:
    """从三个编码组件恢复 [token, feature] 整数，去除流末 padding."""
    if stream.pack_len <= 0 or stream.feature_dim <= 0 or stream.token_count <= 0:
        raise ValueError("invalid bit-packed stream descriptor")
    padded_tokens = stream.token_count + stream.padded_token_count
    if padded_tokens % stream.pack_len:
        raise ValueError("padded token count must be divisible by pack_len")
    pack_count = padded_tokens // stream.pack_len
    field_count = pack_count * stream.feature_dim

    encoded_widths = _unpack_unsigned_fields(
        stream.encode_lengths,
        [stream.encode_length_field_bits] * field_count,
    )
    if any(width > stream.code_value_bits for width in encoded_widths):
        raise ValueError("encode length exceeds code value width")
    encoded_mins = _unpack_unsigned_fields(
        stream.pack_mins,
        [stream.code_value_bits] * field_count,
    )
    pack_mins = [
        (
            _unsigned_to_signed(value, stream.code_value_bits)
            if stream.signed_values
            else value
        )
        for value in encoded_mins
    ]

    payload_widths: List[int] = []
    for width in encoded_widths:
        payload_widths.extend([width] * stream.pack_len)
    payload_values = _unpack_unsigned_fields(stream.payload, payload_widths)

    output = torch.empty(
        (padded_tokens, stream.feature_dim), dtype=torch.int64
    )
    payload_idx = 0
    field_idx = 0
    for pack_idx in range(pack_count):
        for feature_idx in range(stream.feature_dim):
            minimum = pack_mins[field_idx]
            for token_idx in range(stream.pack_len):
                output[
                    pack_idx * stream.pack_len + token_idx, feature_idx
                ] = minimum + payload_values[payload_idx]
                payload_idx += 1
            field_idx += 1
    return output[: stream.token_count]


def bit_pack_decode_kv(
    k_stream: BitPackedCacheStream,
    v_stream: BitPackedCacheStream,
    block_count: int,
    tokens_per_block: int,
) -> torch.Tensor:
    """解码 K/V 并恢复 [block, token, K+V feature] 形状."""
    if block_count <= 0 or tokens_per_block <= 0:
        raise ValueError("block_count and tokens_per_block must be positive")
    expected_tokens = block_count * tokens_per_block
    if k_stream.token_count != expected_tokens or v_stream.token_count != expected_tokens:
        raise ValueError("stream token count does not match requested block shape")
    if k_stream.pack_len != v_stream.pack_len:
        raise ValueError("K/V pack lengths must match")
    k_values = bit_pack_decode(k_stream)
    v_values = bit_pack_decode(v_stream)
    return torch.cat([k_values, v_values], dim=1).reshape(
        block_count, tokens_per_block, -1
    )


@dataclass(frozen=True)
class RepackRoundTripStats:
    """真实重排/metadata/bitstream round-trip 的轻量审计结果."""

    verified_blocks: int
    tokens_per_block: int
    k_stream_bytes: int
    v_stream_bytes: int
    bucket_metadata_bytes: int
    quant_metadata_bytes: int = 0
    joint_dequant_verified: bool = False

    @property
    def total_bytes(self) -> int:
        return (
            self.k_stream_bytes
            + self.v_stream_bytes
            + self.bucket_metadata_bytes
        )

    @property
    def all_encoded_bytes(self) -> int:
        return self.total_bytes + self.quant_metadata_bytes


def verify_repack_bitstream_roundtrip(
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    pack_size: int,
    repack_method: RepackMethod,
    max_blocks: int = 1,
    bucket_count: int = 4,
    bucket_score_method: BucketScoreMethod = BucketScoreMethod.K_SUM,
    k_quant_zero: Optional[torch.Tensor] = None,
    k_quant_scale: Optional[torch.Tensor] = None,
    v_quant_zero: Optional[torch.Tensor] = None,
    v_quant_scale: Optional[torch.Tensor] = None,
    k_zero_point_bits: int = 7,
    v_zero_point_bits: int = 5,
    exponent_bits: int = 4,
    high_precision_zero_point: bool = False,
    fixed_bucket_permutation: Optional[torch.Tensor] = None,
    fixed_repack_metadata: Optional[RepackMetadataStats] = None,
) -> RepackRoundTripStats:
    """对真实量化整数抽样执行重排、字节编码和无损解码.

    输入保持 ``quant_ints`` 的 [B,H,block,token,D] 形状。审计只抽取
    前 ``max_blocks`` 个基础 block，不借用原始 tensor 参与解码；任何整数、
    metadata 或统计字节不一致都会立即抛出异常。
    """
    if k_tensor.shape != v_tensor.shape or k_tensor.ndim != 5:
        raise ValueError("quantized K/V must have matching 5-D shapes")
    if max_blocks <= 0:
        raise ValueError("max_blocks must be positive")
    if isinstance(repack_method, str):
        repack_method = RepackMethod[repack_method]
    if isinstance(bucket_score_method, str):
        bucket_score_method = BucketScoreMethod(bucket_score_method)

    k_blocks = _quant_tensor_to_blocks(k_tensor)
    v_blocks = _quant_tensor_to_blocks(v_tensor)
    verified_blocks = min(max_blocks, k_blocks.shape[0])
    if verified_blocks <= 0:
        raise ValueError("quantized K/V contains no blocks")
    blocks = torch.cat(
        [k_blocks[:verified_blocks], v_blocks[:verified_blocks]], dim=2
    )
    metadata = RepackMetadataStats()
    permutation = None
    if repack_method == RepackMethod.BUCKET:
        if fixed_bucket_permutation is None:
            permutation, metadata = _build_bucket_permutation(
                blocks, bucket_count, bucket_score_method
            )
        else:
            if fixed_repack_metadata is None:
                raise ValueError(
                    "fixed_repack_metadata is required with fixed permutation"
                )
            permutation = fixed_bucket_permutation[:verified_blocks]
            counts = fixed_repack_metadata.bucket_counts[:verified_blocks]
            occupancy_histogram: Dict[int, int] = defaultdict(int)
            for row in counts:
                for count in row:
                    occupancy_histogram[count] += 1
            metadata = RepackMetadataStats(
                bucket_count=fixed_repack_metadata.bucket_count,
                bucket_count_field_bits=fixed_repack_metadata.bucket_count_field_bits,
                bucket_metadata_bits=(
                    verified_blocks
                    * (fixed_repack_metadata.bucket_count - 1)
                    * fixed_repack_metadata.bucket_count_field_bits
                ),
                bucket_counts=counts,
                bucket_occupancy_histogram=dict(sorted(occupancy_histogram.items())),
                bucket_score_method=fixed_repack_metadata.bucket_score_method,
                k_subbucket_count=fixed_repack_metadata.k_subbucket_count,
                v_subbucket_count=fixed_repack_metadata.v_subbucket_count,
            )
        repacked = _apply_token_permutation(blocks, permutation)
    elif repack_method == RepackMethod.NONE:
        repacked = blocks
    elif repack_method == RepackMethod.GREEDY:
        repacked = greedy_repacking(blocks, pack_size)
    elif repack_method == RepackMethod.MEDIAN:
        repacked = median_repacking(blocks)
    else:
        raise ValueError(f"unsupported repack method: {repack_method}")

    metadata_bytes = encode_bucket_metadata(metadata)
    if repack_method == RepackMethod.BUCKET:
        decoded_metadata = decode_bucket_metadata(
            metadata_bytes,
            block_count=verified_blocks,
            tokens_per_block=repacked.shape[1],
            bucket_count=bucket_count,
            score_method=bucket_score_method,
        )
        if decoded_metadata != metadata:
            raise AssertionError("bucket metadata round-trip mismatch")
    elif metadata_bytes:
        raise AssertionError("non-BUCKET repacking produced bucket metadata")

    k_expected_stats, v_expected_stats = bit_pack_stats(repacked, pack_size)
    k_stream, v_stream = bit_pack_encode(repacked, pack_size)
    decoded = bit_pack_decode_kv(
        k_stream,
        v_stream,
        block_count=verified_blocks,
        tokens_per_block=repacked.shape[1],
    )
    expected = repacked.detach().to(device="cpu", dtype=torch.int64)
    if not torch.equal(decoded, expected):
        mismatch_count = int((decoded != expected).sum().item())
        raise AssertionError(
            f"bit-pack round-trip mismatch in {mismatch_count} integer values"
        )
    if k_stream.total_bytes != k_expected_stats.total_bytes:
        raise AssertionError("K bitstream bytes disagree with CR statistics")
    if v_stream.total_bytes != v_expected_stats.total_bytes:
        raise AssertionError("V bitstream bytes disagree with CR statistics")
    if len(metadata_bytes) != metadata.bucket_metadata_bytes:
        raise AssertionError("bucket metadata bytes disagree with CR statistics")

    quant_metadata = (
        k_quant_zero,
        k_quant_scale,
        v_quant_zero,
        v_quant_scale,
    )
    supplied_metadata = [value is not None for value in quant_metadata]
    if any(supplied_metadata) and not all(supplied_metadata):
        raise ValueError("all K/V quantization metadata tensors must be supplied")

    quant_metadata_bytes = 0
    joint_dequant_verified = False
    if all(supplied_metadata):
        if repack_method not in (RepackMethod.NONE, RepackMethod.BUCKET):
            raise ValueError(
                "joint q/metadata audit supports only NONE and BUCKET repacking"
            )

        def selected_metadata(value: torch.Tensor) -> torch.Tensor:
            if value.ndim != 5:
                raise ValueError("quantization metadata must be 5-D")
            return value[:, :, :verified_blocks]

        selected_k_zero = selected_metadata(k_quant_zero)
        selected_k_scale = selected_metadata(k_quant_scale)
        selected_v_zero = selected_metadata(v_quant_zero)
        selected_v_scale = selected_metadata(v_quant_scale)
        if permutation is not None:
            selected_k_zero = apply_quant_metadata_token_permutation(
                selected_k_zero, permutation
            )
            selected_k_scale = apply_quant_metadata_token_permutation(
                selected_k_scale, permutation
            )
            selected_v_zero = apply_quant_metadata_token_permutation(
                selected_v_zero, permutation
            )
            selected_v_scale = apply_quant_metadata_token_permutation(
                selected_v_scale, permutation
            )

        k_metadata_stream = verify_compact_quant_metadata_roundtrip(
            selected_k_zero,
            selected_k_scale,
            k_zero_point_bits,
            exponent_bits,
        )
        v_metadata_stream = verify_compact_quant_metadata_roundtrip(
            selected_v_zero,
            selected_v_scale,
            v_zero_point_bits,
            exponent_bits,
        )
        decoded_k_zero, decoded_k_scale = decode_compact_quant_metadata(
            k_metadata_stream
        )
        decoded_v_zero, decoded_v_scale = decode_compact_quant_metadata(
            v_metadata_stream
        )

        k_feature_dim = k_blocks.shape[2]
        decoded_k_blocks = decoded[:, :, :k_feature_dim]
        decoded_v_blocks = decoded[:, :, k_feature_dim:]

        def blocks_to_tensor(
            decoded_blocks: torch.Tensor, template: torch.Tensor
        ) -> torch.Tensor:
            batch, heads, _, tokens, feature = template.shape
            return decoded_blocks.reshape(
                verified_blocks, tokens, batch, heads, feature
            ).permute(2, 3, 0, 1, 4)

        decoded_k_q = blocks_to_tensor(
            decoded_k_blocks, k_tensor[:, :, :verified_blocks]
        )
        decoded_v_q = blocks_to_tensor(
            decoded_v_blocks, v_tensor[:, :, :verified_blocks]
        )
        decoded_k = dequantize_ints(
            decoded_k_q,
            decoded_k_zero,
            decoded_k_scale,
            high_precision_zero_point,
        )
        decoded_v = dequantize_ints(
            decoded_v_q,
            decoded_v_zero,
            decoded_v_scale,
            high_precision_zero_point,
        )

        expected_k = dequantize_ints(
            k_tensor[:, :, :verified_blocks],
            k_quant_zero[:, :, :verified_blocks],
            k_quant_scale[:, :, :verified_blocks],
            high_precision_zero_point,
        )
        expected_v = dequantize_ints(
            v_tensor[:, :, :verified_blocks],
            v_quant_zero[:, :, :verified_blocks],
            v_quant_scale[:, :, :verified_blocks],
            high_precision_zero_point,
        )
        if permutation is not None:
            expected_k = apply_quant_metadata_token_permutation(
                expected_k, permutation
            )
            expected_v = apply_quant_metadata_token_permutation(
                expected_v, permutation
            )
        expected_k = expected_k.detach().to(device="cpu", dtype=torch.float32)
        expected_v = expected_v.detach().to(device="cpu", dtype=torch.float32)
        if not torch.equal(decoded_k.to(torch.float32), expected_k):
            raise AssertionError("joint K q/metadata dequantization mismatch")
        if not torch.equal(decoded_v.to(torch.float32), expected_v):
            raise AssertionError("joint V q/metadata dequantization mismatch")
        quant_metadata_bytes = (
            k_metadata_stream.total_bytes + v_metadata_stream.total_bytes
        )
        joint_dequant_verified = True

    return RepackRoundTripStats(
        verified_blocks=verified_blocks,
        tokens_per_block=repacked.shape[1],
        k_stream_bytes=k_stream.total_bytes,
        v_stream_bytes=v_stream.total_bytes,
        bucket_metadata_bytes=len(metadata_bytes),
        quant_metadata_bytes=quant_metadata_bytes,
        joint_dequant_verified=joint_dequant_verified,
    )


def bit_pack(blocks: torch.Tensor, pack_len: int) -> Tuple[int, int]:
    """兼容旧调用方,返回按连续bitstream估算的K/V总字节数."""
    k_stats, v_stats = bit_pack_stats(blocks, pack_len)
    return k_stats.total_bytes, v_stats.total_bytes


def bit_pack_detail_rebuttal(
    blocks: torch.Tensor, pack_len: int
) -> Tuple[int, int, int, int, int, int]:
    k_stats, v_stats = bit_pack_stats(blocks, pack_len)
    return (
        k_stats.pack_min_bytes,
        v_stats.pack_min_bytes,
        k_stats.encode_length_bytes,
        v_stats.encode_length_bytes,
        k_stats.payload_bytes,
        v_stats.payload_bytes,
    )


@dataclass(frozen=True)
class PackingAwareCacheStats:
    """单层全局误差预算下的 pack 粒度候选选择统计."""

    total_blocks: int
    total_packs: int
    candidate_different_packs: int
    payload_beneficial_packs: int
    positive_delta_candidates: int
    nonpositive_delta_selected_packs: int
    budget_rejected_beneficial_packs: int
    ceil_selected_packs: int
    nearest_nmse_mean: float
    ceil_nmse_mean: float
    selected_nmse_mean: float
    nearest_sse: float
    ceil_sse: float
    selected_sse: float
    error_budget_sse: float
    used_delta_sse: float
    error_budget_utilization: float
    nearest_payload_bits: int
    ceil_payload_bits: int
    payload_benefit_ceiling_bits: int
    selected_payload_bits: int
    error_budget_violations: int

    @property
    def ceil_selected_rate(self) -> float:
        return self.ceil_selected_packs / self.total_packs if self.total_packs else 0.0

    @property
    def payload_bits_saved(self) -> int:
        return self.nearest_payload_bits - self.selected_payload_bits


def _quant_tensor_to_blocks(tensor: torch.Tensor) -> torch.Tensor:
    """[B,H,block,token,D] -> [block,token,B*H*D]."""
    return tensor.permute(2, 3, 0, 1, 4).flatten(2, 4)


def _packed_payload_bits_per_pack(
    blocks: torch.Tensor, pack_len: int
) -> torch.Tensor:
    """返回 [block, pack] 的 payload bit 数，不计候选无关元数据."""
    if blocks.ndim != 3 or blocks.shape[1] % pack_len != 0:
        raise ValueError("block token count must be divisible by pack_len")
    packs = blocks.to(torch.int64).view(
        blocks.shape[0], -1, pack_len, blocks.shape[2]
    )
    widths = torch.ceil(
        torch.log2(packs.max(dim=2).values - packs.min(dim=2).values + 1)
    ).to(torch.int64)
    return widths.sum(dim=2) * pack_len


def _pack_error_and_signal(
    original: torch.Tensor,
    quant_int: torch.Tensor,
    quant_zero: torch.Tensor,
    quant_scale: torch.Tensor,
    high_precision_zero_point: bool,
    permutation: torch.Tensor,
    pack_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    reconstructed = dequantize_ints(
        quant_int, quant_zero, quant_scale, high_precision_zero_point
    )
    error_blocks = _apply_token_permutation(
        _quant_tensor_to_blocks(
            (original.to(torch.float32) - reconstructed.to(torch.float32)).square()
        ),
        permutation,
    )
    signal_blocks = _apply_token_permutation(
        _quant_tensor_to_blocks(original.to(torch.float32).square()),
        permutation,
    )
    shape = (error_blocks.shape[0], -1, pack_len, error_blocks.shape[2])
    error_sum = error_blocks.reshape(shape).sum(dim=(2, 3))
    signal_sum = signal_blocks.reshape(shape).sum(dim=(2, 3))
    return error_sum, signal_sum


def _record_selected_po2_scales(scales: torch.Tensor) -> None:
    if os.environ.get("PACKKV_GLOBAL_K_STATS", "1") == "0":
        return
    exponents = torch.round(torch.log2(scales.to(torch.float32)))
    unique_ks, counts = torch.unique(exponents.flatten(), return_counts=True)
    for exponent, count in zip(unique_ks.tolist(), counts.tolist()):
        GLOBAL_K_COUNTER[int(exponent)] += count


def packing_aware_quantize_kv(
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    block_size: int,
    pack_size: int,
    k_quant_scale_rel: float,
    v_quant_scale_rel: float,
    k_quant_mode: QuantMode,
    v_quant_mode: QuantMode,
    high_precision_zero_point: bool = False,
    k_error_budget: float = 0.1,
    v_error_budget: float = 0.1,
    bucket_count: int = 4,
    bucket_score_method: BucketScoreMethod = BucketScoreMethod.K_SUM,
):
    """误差约束下的 nearest/ceil 联合软件参考模型.

    nearest K 生成唯一、固定的 Bucket 置换，K/V 共用该置换。重排后的
    每个 pack 计算 ceil 相对 nearest 的 payload 节省和增量 SSE。对有压缩
    收益的候选按 ``saved_bits / positive_delta_sse`` 降序排列，在整层
    ``selected_SSE <= (1 + budget) * nearest_SSE`` 下选择前缀。非正增量
    且有压缩收益的候选总是选择。
    """
    if k_tensor.shape[:3] != v_tensor.shape[:3]:
        raise ValueError("K/V batch, head and sequence dimensions must match")
    if block_size % pack_size:
        raise ValueError("packing-aware mode requires block_size divisible by pack_size")
    if k_error_budget < 0 or v_error_budget < 0:
        raise ValueError("packing-aware error budgets must be non-negative")
    if BucketScoreMethod(bucket_score_method) != BucketScoreMethod.K_SUM:
        raise ValueError("packing-aware reference currently requires bucket_score_method=k_sum")

    def candidates(tensor, scale_rel, quant_mode):
        nearest = quant_ints(
            tensor, block_size, scale_rel, quant_mode,
            high_precision_zero_point, ScaleMethod.PO2_NEAREST,
            record_k_stats=False,
        )
        ceil = quant_ints(
            tensor, block_size, scale_rel, quant_mode,
            high_precision_zero_point, ScaleMethod.PO2_CEIL,
            record_k_stats=False,
        )
        return nearest, ceil

    k_nearest, k_ceil = candidates(k_tensor, k_quant_scale_rel, k_quant_mode)
    v_nearest, v_ceil = candidates(v_tensor, v_quant_scale_rel, v_quant_mode)
    k_nearest_blocks = _quant_tensor_to_blocks(k_nearest[0])
    v_nearest_blocks = _quant_tensor_to_blocks(v_nearest[0])
    nearest_kv_blocks = torch.cat([k_nearest_blocks, v_nearest_blocks], dim=2)
    permutation, repack_metadata = _build_bucket_permutation(
        nearest_kv_blocks, bucket_count, BucketScoreMethod.K_SUM
    )

    original_k = k_tensor.reshape(
        k_tensor.shape[0], k_tensor.shape[1], -1, block_size, k_tensor.shape[3]
    )
    original_v = v_tensor.reshape(
        v_tensor.shape[0], v_tensor.shape[1], -1, block_size, v_tensor.shape[3]
    )

    def choose(original, nearest, ceil, budget):
        nearest_blocks = _apply_token_permutation(
            _quant_tensor_to_blocks(nearest[0]), permutation
        )
        ceil_blocks = _apply_token_permutation(
            _quant_tensor_to_blocks(ceil[0]), permutation
        )
        nearest_cost = _packed_payload_bits_per_pack(nearest_blocks, pack_size)
        ceil_cost = _packed_payload_bits_per_pack(ceil_blocks, pack_size)
        nearest_error, signal = _pack_error_and_signal(
            original, *nearest, high_precision_zero_point, permutation, pack_size
        )
        ceil_error, _ = _pack_error_and_signal(
            original, *ceil, high_precision_zero_point, permutation, pack_size
        )
        payload_beneficial = ceil_cost < nearest_cost
        saved_bits = nearest_cost - ceil_cost
        delta_error = ceil_error - nearest_error
        nonpositive = payload_beneficial & (delta_error <= 0)
        positive_candidates = payload_beneficial & (delta_error > 0)

        nearest_sse = nearest_error.sum()
        error_budget_sse = budget * nearest_sse
        # 非正增量候选可释放误差预算；随后按收益效率选择正增量候选前缀。
        used_nonpositive_delta = delta_error[nonpositive].sum()
        remaining_budget = (error_budget_sse - used_nonpositive_delta).clamp_min(0)
        efficiency = saved_bits.to(torch.float64) / delta_error.to(torch.float64).clamp_min(1e-30)
        candidate_indices = torch.nonzero(positive_candidates.flatten(), as_tuple=False).flatten()
        use_ceil_flat = nonpositive.flatten().clone()
        if candidate_indices.numel():
            order = torch.argsort(
                efficiency.flatten()[candidate_indices], descending=True, stable=True
            )
            ranked_indices = candidate_indices[order]
            cumulative_delta = torch.cumsum(
                delta_error.flatten()[ranked_indices].to(torch.float64), dim=0
            )
            accepted = cumulative_delta <= remaining_budget.to(torch.float64) + 1e-12
            # 前缀策略保持单一效率阈值，便于后续离散为硬件 bucket。
            prefix_len = int(accepted.sum().item())
            if prefix_len:
                use_ceil_flat[ranked_indices[:prefix_len]] = True
        use_ceil = use_ceil_flat.view_as(payload_beneficial)

        def materialize(selection_mask):
            # selection_mask 位于 Bucket 输出顺序；scatter 回原 token 顺序后
            # 选择 quant/zero/scale，fixed permutation 会恢复相同 pack。
            permuted_token_mask = selection_mask.repeat_interleave(
                pack_size, dim=1
            )
            original_token_mask = torch.zeros_like(permuted_token_mask)
            original_token_mask.scatter_(1, permutation, permuted_token_mask)
            token_mask = original_token_mask.view(
                1, 1, original_token_mask.shape[0], block_size, 1
            )
            return tuple(
                torch.where(token_mask, ceil_value, nearest_value)
                for nearest_value, ceil_value in zip(nearest, ceil)
            )

        selected = materialize(use_ceil)
        # payload 收益不一定跨过最终字节边界，也可能被 pack-min/encode-length
        # 抵消。以真实 CR 存储模型复核整层总字节；没有严格减少就回退 nearest，
        # 避免为零压缩收益支付任何额外量化误差。
        selected_blocks = _apply_token_permutation(
            _quant_tensor_to_blocks(selected[0]), permutation
        )
        nearest_stream_stats = _single_cache_bit_pack_stats(
            nearest_blocks.flatten(0, 1), pack_size, padded_token_count=0
        )
        selected_stream_stats = _single_cache_bit_pack_stats(
            selected_blocks.flatten(0, 1), pack_size, padded_token_count=0
        )
        if use_ceil.any() and (
            selected_stream_stats.total_bytes >= nearest_stream_stats.total_bytes
        ):
            use_ceil = torch.zeros_like(use_ceil)
            selected = nearest

        selected_error = torch.where(use_ceil, ceil_error, nearest_error)
        nearest_nmse = nearest_error / signal.clamp_min(1e-12)
        ceil_nmse = ceil_error / signal.clamp_min(1e-12)
        selected_nmse = selected_error / signal.clamp_min(1e-12)
        selected_cost = torch.where(use_ceil, ceil_cost, nearest_cost)
        nearest_scale_blocks = _apply_token_permutation(
            _quant_tensor_to_blocks(nearest[2]), permutation
        )
        ceil_scale_blocks = _apply_token_permutation(
            _quant_tensor_to_blocks(ceil[2]), permutation
        )
        scale_diff = (nearest_scale_blocks != ceil_scale_blocks).reshape(
            nearest_scale_blocks.shape[0], -1, pack_size, nearest_scale_blocks.shape[2]
        ).any(dim=(2, 3))
        selected_sse = selected_error.sum()
        used_delta_sse = selected_sse - nearest_sse
        tolerance = 1e-5 * nearest_sse.abs().clamp_min(1.0)
        violations = selected_sse > (1.0 + budget) * nearest_sse + tolerance
        potential_cost = torch.where(payload_beneficial, ceil_cost, nearest_cost)
        positive_budget = float(error_budget_sse.item())
        utilization = (
            max(0.0, float(used_delta_sse.item())) / positive_budget
            if positive_budget > 0
            else 0.0
        )
        stats = PackingAwareCacheStats(
            total_blocks=int(permutation.shape[0]),
            total_packs=int(use_ceil.numel()),
            candidate_different_packs=int(scale_diff.sum().item()),
            payload_beneficial_packs=int(payload_beneficial.sum().item()),
            positive_delta_candidates=int(positive_candidates.sum().item()),
            nonpositive_delta_selected_packs=int(
                (nonpositive & use_ceil).sum().item()
            ),
            budget_rejected_beneficial_packs=int(
                (payload_beneficial & ~use_ceil).sum().item()
            ),
            ceil_selected_packs=int(use_ceil.sum().item()),
            nearest_nmse_mean=float(nearest_nmse.mean().item()),
            ceil_nmse_mean=float(ceil_nmse.mean().item()),
            selected_nmse_mean=float(selected_nmse.mean().item()),
            nearest_sse=float(nearest_sse.item()),
            ceil_sse=float(ceil_error.sum().item()),
            selected_sse=float(selected_sse.item()),
            error_budget_sse=positive_budget,
            used_delta_sse=float(used_delta_sse.item()),
            error_budget_utilization=utilization,
            nearest_payload_bits=int(nearest_cost.sum().item()),
            ceil_payload_bits=int(ceil_cost.sum().item()),
            payload_benefit_ceiling_bits=int(potential_cost.sum().item()),
            selected_payload_bits=int(selected_cost.sum().item()),
            error_budget_violations=int(violations.item()),
        )
        _record_selected_po2_scales(selected[2])
        return selected, stats

    k_selected, k_stats = choose(original_k, k_nearest, k_ceil, k_error_budget)
    v_selected, v_stats = choose(original_v, v_nearest, v_ceil, v_error_budget)
    return (
        *k_selected,
        *v_selected,
        permutation,
        repack_metadata,
        k_stats,
        v_stats,
    )


def repack_and_encode(
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    pack_size: int,
    repack_method: RepackMethod,
    before_and_after_repacking=None,
    return_stats: bool = False,
    bucket_count: int = 4,
    bucket_score_method: BucketScoreMethod = BucketScoreMethod.COMBINED_SUM,
    return_repack_metadata: bool = False,
    fixed_bucket_permutation: Optional[torch.Tensor] = None,
    fixed_repack_metadata: Optional[RepackMetadataStats] = None,
):
    """执行不同的重排算法, 并对比重排前后的收益与代价"""
    k_blocks = k_tensor.permute(2, 3, 0, 1, 4).flatten(2, 4)
    v_blocks = v_tensor.permute(2, 3, 0, 1, 4).flatten(2, 4)
    blocks = torch.cat([k_blocks, v_blocks], dim=2)
    k_stats_pre, v_stats_pre = bit_pack_stats(blocks, pack_size)
    repack_metadata = RepackMetadataStats()

    before_and_after_ = [blocks, None]
    if repack_method == RepackMethod.GREEDY:
        blocks = greedy_repacking(blocks, pack_size)
    elif repack_method == RepackMethod.MEDIAN:
        blocks = median_repacking(blocks)
    elif repack_method == RepackMethod.BUCKET:
        if fixed_bucket_permutation is None:
            blocks, repack_metadata = bucket_repacking(
                blocks,
                num_buckets=bucket_count,
                score_method=bucket_score_method,
                return_metadata=True,
            )
        else:
            if fixed_repack_metadata is None:
                raise ValueError("fixed_repack_metadata is required with fixed permutation")
            blocks = _apply_token_permutation(blocks, fixed_bucket_permutation)
            repack_metadata = fixed_repack_metadata
    elif repack_method == RepackMethod.NONE:
        pass
    else:
        raise ValueError(
            f"repack_method must be one of {RepackMethod.__members__.keys()}"
        )

    before_and_after_[1] = blocks

    if before_and_after_repacking is not None:
        before_and_after_repacking.append(before_and_after_)

    k_stats_aft, v_stats_aft = bit_pack_stats(blocks, pack_size)

    if return_stats:
        if return_repack_metadata:
            return (
                k_stats_pre,
                v_stats_pre,
                k_stats_aft,
                v_stats_aft,
                repack_metadata,
            )
        return k_stats_pre, v_stats_pre, k_stats_aft, v_stats_aft
    return (
        k_stats_pre.total_bytes,
        v_stats_pre.total_bytes,
        k_stats_aft.total_bytes,
        v_stats_aft.total_bytes,
    )


def repack_and_encode_detail_rebuttal(
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    pack_size: int,
    repack_method: RepackMethod = RepackMethod.MEDIAN,
) -> Tuple[int, int, int, int, int, int]:
    k_blocks = k_tensor.permute(2, 3, 0, 1, 4).flatten(2, 4)
    v_blocks = v_tensor.permute(2, 3, 0, 1, 4).flatten(2, 4)
    blocks = torch.cat([k_blocks, v_blocks], dim=2)
    # blocks = median_repacking(blocks)
    if repack_method == RepackMethod.GREEDY:
        blocks = greedy_repacking(blocks, pack_size)
    elif repack_method == RepackMethod.MEDIAN:
        blocks = median_repacking(blocks)
    elif repack_method == RepackMethod.BUCKET:
        blocks = bucket_repacking(blocks, num_buckets=4)
    elif repack_method == RepackMethod.NONE:
        pass
    else:
        raise ValueError(
            f"repack_method must be one of {RepackMethod.__members__.keys()}"
        )
    (
        k_zero_point_size,
        v_zero_point_size,
        k_encode_len_size,
        v_encode_len_size,
        k_pack_size,
        v_pack_size,
    ) = bit_pack_detail_rebuttal(blocks, pack_size)

    return (
        k_zero_point_size,
        v_zero_point_size,
        k_encode_len_size,
        v_encode_len_size,
        k_pack_size,
        v_pack_size,
    )


def repack_throughput_detail_rebuttal(
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    pack_size: int,
) -> Tuple[float, float]:
    """性能(延迟)对比函数"""
    k_blocks = k_tensor.permute(2, 3, 0, 1, 4).flatten(2, 4)
    v_blocks = v_tensor.permute(2, 3, 0, 1, 4).flatten(2, 4)
    blocks_ = torch.cat([k_blocks, v_blocks], dim=2)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    blocks = greedy_repacking(blocks_, pack_size)
    end.record()
    torch.cuda.synchronize()
    greedy_time = start.elapsed_time(end)

    start.record()
    blocks = median_repacking(blocks_)
    end.record()
    torch.cuda.synchronize()
    median_time = start.elapsed_time(end)

    return greedy_time, median_time


def entropy(tensor):
    """计算张量中数值分布的信息熵"""
    # 统计了每个数字出现的频次
    values, counts = torch.unique(tensor, return_counts=True)
    # 算出每个数字出现的概率p
    probs = counts.float() / counts.sum()
    # 套用信息熵公式
    entropy = -torch.sum(probs * torch.log2(probs))
    return entropy


QUANT_DIM = {
    QuantMode.BlockQuant.value: [1, 3, 4],
    QuantMode.ChannelQuant.value: [3],
    QuantMode.TokenQuant.value: [1, 4],
    QuantMode.VectorQuant.value: [4],
}
