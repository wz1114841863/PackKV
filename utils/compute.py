import torch
from typing import Tuple, Optional, List
from enum import Enum
import math


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
    """将动态增长的 Cache 拼接 (safe_cat) 起来，并按照 block_size 进行切分。
    只有凑够了一个完整的 Block（且排除了 recent_size 即最近的无需压缩的高精度 Token），才会被送入后续的量化流程。
    这模拟了硬件执行流中，数据从片上 SRAM (Buffer) 满载后，被压缩写回大容量 DRAM 的过程。"""
    buffer = safe_cat(buffer, new_tensor, dim)
    len_ = buffer.shape[dim]
    res_num = len_ % block_size
    to_compress_block_num = (len_ + block_size - res_num - recent_size) // block_size
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert (
        tensor.shape[2] % block_size == 0
    ), "Tensor shape is not divisible by block size"
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

    quant_scale = (max_val - min_val) * quant_scale_rel
    quant_scale = torch.clamp(quant_scale, min=1e-5)

    if high_precision_zero_point:
        # -min_val, /scale
        # quant_scale_rel控制的是量化网格(Grid)的步长.由于大模型不同层/不同 Token 激活值的极差($X_{max} - X_{min}$)浮动非常大,
        # 不能用一个固定的绝对数值来做步长.因此,算法采用极差乘以一个相对比例 quant_scale_rel 来动态决定当前数据块的量化步长.
        # 这个值越大,压缩率越高,精度损失越大.

        value_quant = ((tensor - min_val) / quant_scale).round()
    else:
        # /scale, -min_int, 将零点偏移量本身也量化成整数
        # quant_scale = (max_val - min_val) * quant_scale_rel
        min_int = (min_val / quant_scale).round()
        value_quant = (tensor / quant_scale).round() - min_int
        min_val = min_int

    return value_quant, min_val, quant_scale


def quant_ints_2k(
    tensor: torch.Tensor,
    block_size: int,
    quant_scale_rel: float,
    quant_mode: QuantMode,
    high_precision_zero_point: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    硬件友好的 2^k 移位量化器 (集成容器感知逻辑)
    保持与原版 quant_ints 完全一致的输入参数和返回值结构.
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
    max_int_init = range_ / scale_init
    target_bits = torch.clamp(torch.ceil(torch.log2(max_int_init + 1)), min=1.0)

    # (2) 容器拉伸:计算该位宽下能把容器撑满的理想 Scale
    c_max = (2**target_bits) - 1
    scale_ideal = range_ / c_max

    # (3) 二次幂逼近 (保精度 Round 策略)
    k = torch.round(torch.log2(scale_ideal))
    quant_scale = torch.pow(2.0, k)  # 最终 Scale = 2^k
    # ==========================================

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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """伪量化, 计算量化过程带来的误差, 用于算法维度的验证和补偿"""
    to_compress, in_buffer = cut_tensor(
        buffer, new_tensor, block_size, recent_size, dim=2
    )

    if to_compress is not None:
        # quant_int, quant_zero, quant_scale = quant_ints(
        quant_int, quant_zero, quant_scale = quant_ints_2k(
            to_compress,
            block_size,
            quant_scale_rel,
            quant_mode,
            high_precision_zero_point,
        )
        if high_precision_zero_point:
            # -min_val, /scale
            to_compress = quant_int * quant_scale + quant_zero
        else:
            # /scale, -min_int
            to_compress = (quant_int + quant_zero) * quant_scale
        to_compress = to_compress.reshape(
            to_compress.shape[0], to_compress.shape[1], -1, to_compress.shape[4]
        )

    return safe_cat(error_cache, to_compress, dim=2), in_buffer


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


def bucket_repacking(blocks: torch.Tensor, num_buckets: int = 4) -> torch.Tensor:
    """面向硬件的动态阈值分桶重排算法
    与硬件结合, 简化排序算法不按中位数绝对排序, 而是按阈值把 Token 扔进几个桶里,
    以此来模拟硬件中低延迟的比较器路由逻辑.
    参数:
        blocks: 输入的张量块,形状为 [B, N, D]
                (B: 批次, N: Token数量/Block大小, D: 特征维度, K和V拼接)
        num_buckets: 硬件中设计的 FIFO 桶的数量,默认为 4
    """
    B, N, D = blocks.shape
    half_vec_len = D // 2
    # 依旧基于V向量提取代表值
    v_part = blocks[:, :, half_vec_len:]

    # 为了和原版 Baseline 控制变量,这里先保留 median.
    # 实际在RTL实现时, 可以把这里的median换成 mean(均值在硬件中用加法树实现极简单)
    feature_vals = torch.median(v_part, dim=2).values  # 形状: [B, N]

    # 动态确定硬件阈值边界: 获取这批Token特征的最大值和最小值
    b_min = feature_vals.min(dim=1, keepdim=True).values
    b_max = feature_vals.max(dim=1, keepsim=True).values

    # 计算每个桶的宽度/步长
    step = (b_max - b_min) / num_buckets
    setp = torch.clamp(step, min=1e-6)  # 避免除以0

    # 将特征值映射到0到(num_buckets-1)的桶索引
    bucket_ids = torch.floor((feature_vals - b_min) / step).long()
    # 处理边界移除, 确保最大值落在最后一个桶里
    bucket_ids = torch.clamp(bucket_ids, min=0, max=num_buckets - 1)

    # 模拟硬件的FIFO路由
    # 硬件中数据会根据 bucket_id 直接通过交叉开关(Crossbar)掉进 4 个不同的FIFO里,不需要排序.
    # 但在 PyTorch 里,为了把相同 bucket_id 的 Token 聚在内存的相邻位置,
    # 我们只能"借用"一下 argsort.你要清楚,这个 sort 在转 RTL 时是完全不存在的!
    _, sorted_indices = torch.sort(bucket_ids, dim=1, descending=True)

    # 完成物理位置重组
    sorted_indices_expanded = sorted_indices.unsqueeze(2).expand(B, N, D)
    repacked_blocks = torch.gather(blocks, 1, sorted_indices_expanded)

    return repacked_blocks


def hardware_bucket_repacking(
    blocks: torch.Tensor, num_main_buckets: int = 4, clip_val: float = 5.0
) -> torch.Tensor:
    """
    行为级硬件仿真:带异常值截断旁路 (Bypass) 的分桶重排

    参数:
        blocks: 输入张量 [B, N, D]
        num_main_buckets: 常规数据的分类桶数量 (硬件中主流水线的 FIFO 数量)
        clip_val: 经验截断阈值,比如 5.0.超过这个绝对值的 Token 走异常值通道.
    """
    B, N, D = blocks.shape
    half_vec_len = D // 2

    # 1. 特征提取 (硬件中的加法树或中位数选择网络)
    v_part = blocks[:, :, half_vec_len:]
    feature_vals = torch.median(v_part, dim=2).values  # [B, N]

    # 2. 硬件固定的步长 (写死在 RTL 里的常数,不需要动态除法器)
    # 例如:范围是 [-5, 5],分成 4 个主桶,每个桶的宽度 (步长) 是 2.5
    step = (2.0 * clip_val) / num_main_buckets

    # 3. 硬件比较器阵列打 Tag
    # 算出基础的 bucket ID.利用 + clip_val 将 [-5, 5] 平移到 [0, 10]
    raw_bucket_ids = torch.floor((feature_vals + clip_val) / step).long()

    # 4. 模拟硬件的"异常值旁路 (Bypass)"逻辑
    # 我们为硬件规划 (num_main_buckets + 2) 个桶:
    # ID = 0: 负向异常值桶 (小于 -5.0)
    # ID = 1 到 num_main_buckets: 主数据流水线
    # ID = num_main_buckets + 1: 正向异常值桶 (大于 5.0)

    # 给主数据腾出 ID 空间 (向右平移 1 位)
    bucket_ids = raw_bucket_ids + 1

    # 硬件限幅器 (Limiter / Saturation Logic):
    # 把越界的值死死卡在 0 和 num_main_buckets + 1
    bucket_ids = torch.clamp(bucket_ids, min=0, max=num_main_buckets + 1)

    # 5. 模拟物理聚合 (在硬件中是根据 ID 送入对应的 SRAM Bank)
    _, sorted_indices = torch.sort(bucket_ids, dim=1, descending=True)
    sorted_indices_expanded = sorted_indices.unsqueeze(2).expand(B, N, D)
    repacked_blocks = torch.gather(blocks, 1, sorted_indices_expanded)

    return repacked_blocks


def bit_pack(blocks: torch.Tensor, pack_len: int) -> Tuple[int, int]:
    """评估算法理论压缩率"""
    # 将传入的张量划分为K和V, 切分成大小为pack_len的组
    # 并计算每个组的最大值和最小值
    half_vec_len = blocks.shape[2] // 2
    blocks = blocks.flatten(0, 1).to(torch.int64)
    k_blocks = blocks[:, :half_vec_len]
    v_blocks = blocks[:, half_vec_len:]

    k_packs = k_blocks.view(-1, pack_len, half_vec_len)
    v_packs = v_blocks.view(-1, pack_len, half_vec_len)

    k_bit_len = math.ceil(math.log2(k_packs.unique().numel()))
    v_bit_len = math.ceil(math.log2(v_packs.unique().numel()))
    # 找出基础值
    k_pack_mins = k_packs.min(dim=1).values
    k_pack_maxs = k_packs.max(dim=1).values
    v_pack_mins = v_packs.min(dim=1).values
    v_pack_maxs = v_packs.max(dim=1).values

    # 计算有效载荷, 即存下Pack内数值所需的基础比特
    # 组内极差 = k_pack_maxs - k_pack_mins
    # 所需位宽 = ceil(log2(极差 + 1))
    k_pack_bit_num = (
        torch.ceil(torch.log2(k_pack_maxs - k_pack_mins + 1)).to(torch.int64).sum()
        * pack_len
    )
    v_pack_bit_num = (
        torch.ceil(torch.log2(v_pack_maxs - v_pack_mins + 1)).to(torch.int64).sum()
        * pack_len
    )

    k_pack_bit_num = torch.clamp(k_pack_bit_num, min=2.0)
    v_pack_bit_num = torch.clamp(v_pack_bit_num, min=2.0)

    # 总比特数 += 基础值的数量 * (基础值所需的比特 + 编码头所需的比特)
    k_pack_bit_num += k_pack_mins.numel() * (
        k_bit_len + math.ceil(math.log2(k_bit_len + 1))
    )
    v_pack_bit_num += v_pack_mins.numel() * (
        v_bit_len + math.ceil(math.log2(v_bit_len + 1))
    )

    return k_pack_bit_num.item() // 8, v_pack_bit_num.item() // 8


def bit_pack_detail_rebuttal(
    blocks: torch.Tensor, pack_len: int
) -> Tuple[int, int, int, int, int, int]:
    half_vec_len = blocks.shape[2] // 2
    blocks = blocks.flatten(0, 1).to(torch.int64)
    k_blocks = blocks[:, :half_vec_len]
    v_blocks = blocks[:, half_vec_len:]

    k_packs = k_blocks.view(-1, pack_len, half_vec_len)
    v_packs = v_blocks.view(-1, pack_len, half_vec_len)

    k_bit_len = math.ceil(math.log2(k_packs.unique().numel()))
    v_bit_len = math.ceil(math.log2(v_packs.unique().numel()))

    k_pack_mins = k_packs.min(dim=1).values
    k_pack_maxs = k_packs.max(dim=1).values
    v_pack_mins = v_packs.min(dim=1).values
    v_pack_maxs = v_packs.max(dim=1).values

    # 计算真正的有效载荷 (Payload，即打包后的差值)
    k_pack_bit_num = (
        torch.ceil(torch.log2(k_pack_maxs - k_pack_mins + 1)).to(torch.int64).sum()
        * pack_len
    )
    v_pack_bit_num = (
        torch.ceil(torch.log2(v_pack_maxs - v_pack_mins + 1)).to(torch.int64).sum()
        * pack_len
    )
    # 计算 K 和 V 的零点元数据开销 (Zero-point)
    k_zero_point_bit_num = k_pack_mins.numel() * k_bit_len
    # 计算 K 和 V 的位宽字典开销 (Encode-length)
    k_encode_len_bit_num = k_pack_mins.numel() * math.ceil(math.log2(k_bit_len + 1))
    v_zero_point_bit_num = v_pack_mins.numel() * v_bit_len
    v_encode_len_bit_num = v_pack_mins.numel() * math.ceil(math.log2(v_bit_len + 1))

    return (
        k_zero_point_bit_num // 8,
        v_zero_point_bit_num // 8,
        k_encode_len_bit_num // 8,
        v_encode_len_bit_num // 8,
        k_pack_bit_num.item() // 8,
        v_pack_bit_num.item() // 8,
    )


def repack_and_encode(
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    pack_size: int,
    repack_method: RepackMethod,
    before_and_after_repacking=None,
) -> Tuple[int, int, int, int]:
    """执行不同的重排算法, 并对比重排前后的收益与代价"""
    k_blocks = k_tensor.permute(2, 3, 0, 1, 4).flatten(2, 4)
    v_blocks = v_tensor.permute(2, 3, 0, 1, 4).flatten(2, 4)
    blocks = torch.cat([k_blocks, v_blocks], dim=2)
    k_size_pre, v_size_pre = bit_pack(blocks, pack_size)

    before_and_after_ = [blocks, None]
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

    before_and_after_[1] = blocks

    if before_and_after_repacking is not None:
        before_and_after_repacking.append(before_and_after_)

    k_size_aft, v_size_aft = bit_pack(blocks, pack_size)

    return k_size_pre, v_size_pre, k_size_aft, v_size_aft


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
