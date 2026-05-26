import torch
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.compute import (
    quant_ints,
    QuantMode,
    median_repacking,
    repack_and_encode,
    RepackMethod,
)


def debug_pipeline():
    # 1. 模拟生成一个 [Batch=1, Heads=32, SeqLen=128, HeadDim=128] 的 V Cache
    # 使用 randn 模拟真实的浮点数据
    dummy_v_cache = torch.randn(1, 32, 128, 128)

    print("--- 步骤 1: 量化 ---")
    block_size = 16
    # 模拟 Token 级别的量化
    quant_int, min_val, scale = quant_ints(
        tensor=dummy_v_cache,
        block_size=block_size,
        quant_scale_rel=1.0,
        quant_mode=QuantMode.TokenQuant,
    )
    print(f"量化后整数张量的形状: {quant_int.shape}")
    # 建议在这里打断点，查看 quant_int 内部的最大最小值

    print("\n--- 步骤 2: 重排与位打包大小评估 ---")
    # 模拟 repack_and_encode_detail_rebuttal 中的重排前后的空间对比
    # 注意: repack_and_encode 需要输入特定 shape，这里用简化的 block 直接测试 median_repacking

    # 假设我们取出一个 Head 的一个 Block 进行观察 (Batch=1, N=16, Dim=128)
    sample_block = quant_int[0, 0, 0, :, :]
    # 扩展维度以适应 median_repacking 的 (B, N, D) 输入
    sample_block = sample_block.unsqueeze(0)

    print(f"重排前 Block 的前两个 Token 值: \n{sample_block[0, :2, :5]}")

    repacked_block = median_repacking(sample_block)

    print(f"重排后 Block 的前两个 Token 值: \n{repacked_block[0, :2, :5]}")
    # 建议在这里打断点：比较 sample_block 和 repacked_block，你会发现相似的数值被聚集在一起了


if __name__ == "__main__":
    debug_pipeline()
