import torch
from transformers import Cache
from typing import Optional, Dict, Any, Tuple
from utils.compute import (
    apply_rotary_pos_emb_single,
    safe_cat,
    quant_error,
    QuantMethod,
)
from utils.util import JumpOutException


class PackKVCacheConfigStatic:
    config = None
    extract_cache = None


def mad_based_center(values, dim, k=3, keepdim=True):
    # 找中位数, 中位数不受极端值影响
    median = values.median(dim=dim, keepdim=True).values
    # 计算MAD(绝对中位差)：计算每个数偏离中位数的距离，然后再取中位数
    mad = (values - median).abs().median(dim=dim, keepdim=True).values
    # 如果某个数值偏离中位数的距离，大于3倍()k=3)的MAD
    # 就认为它是异常值，将其mask掉
    mask = (values - median).abs() < k * mad
    if keepdim:
        mask_sum = mask.sum(dim=dim, keepdim=True)
        safe_mask = mask_sum > 0
        # 只把正常值加起来(values * mask), 除以正常值的个数(mask_sum)
        masked_mean = (values * mask).sum(dim=dim, keepdim=True) / mask_sum.clamp(min=1)
        # 如果一整行全是异常值, 就退回使用一开始算出的中位数
        masked_mean = torch.where(safe_mask, masked_mean, median)
    else:
        mask_sum = mask.sum(dim=dim)
        safe_mask = mask_sum > 0
        masked_mean = (values * mask).sum(dim=dim) / mask_sum.clamp(min=1)
        masked_mean = torch.where(safe_mask, masked_mean, median.squeeze(dim))
    return masked_mean


class PackKVCachePytorchQuant(Cache):
    round_ = 0

    def __init__(self, batch_size, head_num, head_dim, layer_num):
        PackKVCachePytorchQuant.round_ += 1
        super().__init__()
        self.batch_size = batch_size
        self.head_num = head_num
        self.head_dim = head_dim
        # 缓冲的双层结构, 每一层的K, V都分为compressed和buffer两部分
        self.compressed_k_cache = [None] * layer_num
        self.compressed_v_cache = [None] * layer_num
        self.k_cache_buffer = [None] * layer_num
        self.v_cache_buffer = [None] * layer_num
        self.coss = [None] * layer_num
        self.sins = [None] * layer_num
        self.k_avg = [None] * layer_num

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if PackKVCacheConfigStatic.extract_cache is not None:
            assert (
                not PackKVCacheConfigStatic.config.enable_quant
            ), "Disable quant to enable cache collect."
        if PackKVCacheConfigStatic.config is None:
            raise ValueError(
                "TComCacheConfigStatic.config is not set. Please set it before using TComCache."
            )
        if PackKVCacheConfigStatic.config.enable_quant is False:
            return self.update_disable(
                key_states, value_states, layer_idx, cache_kwargs
            )
        # 在缓存之前, 先将RoPE应用到Key
        cos, sin = cache_kwargs["cos"], cache_kwargs["sin"]
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin)

        self.compressed_k_cache[layer_idx], self.k_cache_buffer[layer_idx] = (
            quant_error(
                self.compressed_k_cache[layer_idx],
                self.k_cache_buffer[layer_idx],
                key_states,  # 新来的 Token
                PackKVCacheConfigStatic.config.block_size,
                PackKVCacheConfigStatic.config.buffer_size,
                PackKVCacheConfigStatic.config.k_quant_scale_rel,
                PackKVCacheConfigStatic.config.quant_method.value[0],
                PackKVCacheConfigStatic.config.high_precision_zero_point,
            )
        )

        self.compressed_v_cache[layer_idx], self.v_cache_buffer[layer_idx] = (
            quant_error(
                self.compressed_v_cache[layer_idx],
                self.v_cache_buffer[layer_idx],
                value_states,
                PackKVCacheConfigStatic.config.block_size,
                PackKVCacheConfigStatic.config.buffer_size,
                PackKVCacheConfigStatic.config.v_quant_scale_rel,
                PackKVCacheConfigStatic.config.quant_method.value[1],
                PackKVCacheConfigStatic.config.high_precision_zero_point,
            )
        )
        # + self.k_avg[layer_idx]
        # 通过 safe_cat 将压缩后的老缓存和未压缩的新缓冲区在序列维度（dim=2）上拼接起来，返回给注意力层进行点积计算。
        return safe_cat(
            self.compressed_k_cache[layer_idx], self.k_cache_buffer[layer_idx], dim=2
        ), safe_cat(
            self.compressed_v_cache[layer_idx], self.v_cache_buffer[layer_idx], dim=2
        )

    def update_disable(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = cache_kwargs["cos"], cache_kwargs["sin"]
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin)

        self.k_cache_buffer[layer_idx] = safe_cat(
            self.k_cache_buffer[layer_idx], key_states, dim=2
        )
        self.v_cache_buffer[layer_idx] = safe_cat(
            self.v_cache_buffer[layer_idx], value_states, dim=2
        )

        if PackKVCacheConfigStatic.extract_cache is not None:
            if (
                key_states.shape[2] != 1
                and PackKVCachePytorchQuant.round_
                > PackKVCacheConfigStatic.extract_cache.collect_round
            ):
                raise JumpOutException(
                    "\nExtracting cache and new generation round has been detected, jump out to process cache."
                )
            if (
                PackKVCachePytorchQuant.round_ - 1
                not in PackKVCacheConfigStatic.extract_cache.key_caches
            ):
                PackKVCacheConfigStatic.extract_cache.key_caches[
                    PackKVCachePytorchQuant.round_ - 1
                ] = {}
                PackKVCacheConfigStatic.extract_cache.value_caches[
                    PackKVCachePytorchQuant.round_ - 1
                ] = {}
            PackKVCacheConfigStatic.extract_cache.key_caches[
                PackKVCachePytorchQuant.round_ - 1
            ][layer_idx - 1] = self.k_cache_buffer[layer_idx]
            PackKVCacheConfigStatic.extract_cache.value_caches[
                PackKVCachePytorchQuant.round_ - 1
            ][layer_idx - 1] = self.v_cache_buffer[layer_idx]
            # print(f"collected KV cache size: {}")

        return self.k_cache_buffer[layer_idx], self.v_cache_buffer[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """总序列长度 = 压缩缓存的长度 + 缓冲区的长度"""
        if PackKVCacheConfigStatic.config is None:
            raise ValueError(
                "TComCacheConfigStatic.config is not set. Please set it before using TComCache."
            )
        if PackKVCacheConfigStatic.config.enable_quant is False:
            if layer_idx is None:
                return self.k_cache_buffer[0].shape[2]
            else:
                if self.k_cache_buffer[layer_idx] is None:
                    return 0
                return self.k_cache_buffer[layer_idx].shape[2]

        if layer_idx is None:
            self.compressed_k_cache[0].shape[2] + self.k_cache_buffer[0].shape[2]
        else:
            if (
                self.compressed_k_cache[layer_idx] is None
                and self.k_cache_buffer[layer_idx] is None
            ):
                return 0
            if self.compressed_k_cache[layer_idx] is None:
                return self.k_cache_buffer[layer_idx].shape[2]
            if self.k_cache_buffer[layer_idx] is None:
                return self.compressed_k_cache[layer_idx].shape[2]
            return (
                self.compressed_k_cache[layer_idx].shape[2]
                + self.k_cache_buffer[layer_idx].shape[2]
            )
