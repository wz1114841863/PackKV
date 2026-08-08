import torch
import os
import math
import json
from importlib import metadata
from datetime import datetime
from datasets import load_dataset
from torch import nn
from typing import Tuple, List
from models.cache.packkv_quant import PackKVCacheConfigStatic, PackKVCachePytorchQuant
from models.llama import LlamaForCausalLM
from models.phi import Phi3ForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
from lm_eval import evaluator
from models.qwen3 import Qwen3ForCausalLM
from models.mistral import MistralForCausalLM
from utils.compute import (
    BucketScoreMethod,
    QuantMode,
    QuantMethod,
    quant_ints,
    repack_and_encode,
    repack_and_encode_detail_rebuttal,
    repack_throughput_detail_rebuttal,
    quant_ints_throughput,
    ScaleMethod,
    dequantize_ints,
    packing_aware_quantize_kv,
    verify_repack_bitstream_roundtrip,
)
from utils.config import PackKVCacheConfig, ExtractCacheConfig
from utils.lm_eval_warp import LMEvalWrapper
from utils.profiling_func import profile_func
from utils.util import JumpOutException
from packkv_cuda import (
    k_encode_cpu,
    k_decode_cpu,
    kq_mat_vec_mul,
    v_encode_cpu,
    v_decode_cpu,
    wv_mat_vec_mul,
    fused_kq,
    fused_wv,
)

MODEL_CLASS_MAP = {
    "JackFram/llama-160m": LlamaForCausalLM,
    "huggyllama/llama-7b": LlamaForCausalLM,
    "meta-llama/Llama-2-7b-hf": LlamaForCausalLM,
    "meta-llama/Llama-2-13b-hf": LlamaForCausalLM,
    "meta-llama/Llama-3.1-8B": LlamaForCausalLM,
    "meta-llama/Meta-Llama-3-8B": LlamaForCausalLM,
    "huggyllama/llama-13b": LlamaForCausalLM,
    "Qwen/Qwen3-8B": Qwen3ForCausalLM,
    "Qwen/Qwen3-4B": Qwen3ForCausalLM,
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": Qwen3ForCausalLM,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": LlamaForCausalLM,
    "mistralai/Ministral-8B-Instruct-2410": MistralForCausalLM,
    "NousResearch/Meta-Llama-3-8B": LlamaForCausalLM,
    "NousResearch/Meta-Llama-3.1-8B": LlamaForCausalLM,
    "mistralai/Ministral-3-8B-Instruct-2512": MistralForCausalLM,
    "microsoft/phi-4": Phi3ForCausalLM,
}


def _package_version(distribution_name: str) -> str:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _accuracy_run_metadata(
    config: PackKVCacheConfig,
    task: str,
    batch_size: int | str,
    limit: int | None,
    effective_model_dtype: str,
) -> dict:
    """构造独立于 lm-eval results.json 的 PackKV 实验元数据."""
    quant_method = (
        getattr(config, "quant_method", None) if config.enable_quant else None
    )
    quant_modes = quant_method.value if quant_method is not None else (None, None)
    repack_method = getattr(config, "repack_method", None)
    scale_method = getattr(config, "scale_method", ScaleMethod.CONTINUOUS)
    bucket_score_method = getattr(
        config,
        "bucket_score_method",
        BucketScoreMethod.COMBINED_SUM,
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": config.model_name,
        "task": task,
        "limit": limit,
        "batch_size": batch_size,
        "enable_quant": bool(config.enable_quant),
        "quant_method": quant_method.name if quant_method is not None else None,
        "k_quant_mode": quant_modes[0].value if quant_modes[0] is not None else None,
        "v_quant_mode": quant_modes[1].value if quant_modes[1] is not None else None,
        "scale_method": scale_method.value,
        "k_scale": getattr(config, "k_quant_scale_rel", None),
        "v_scale": getattr(config, "v_quant_scale_rel", None),
        "repack_method": (
            repack_method.name if repack_method is not None else None
        ),
        "high_precision_zero_point": bool(
            getattr(config, "high_precision_zero_point", False)
        ),
        "block_size": getattr(config, "block_size", None),
        "buffer_size": getattr(config, "buffer_size", None),
        "pack_size": getattr(config, "pack_size", None),
        "bucket_count": getattr(config, "bucket_count", None),
        "bucket_score_method": bucket_score_method.value,
        "k_error_budget": getattr(config, "k_error_budget", None),
        "v_error_budget": getattr(config, "v_error_budget", None),
        "effective_model_dtype": effective_model_dtype,
        "software_versions": {
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "lm_eval": _package_version("lm_eval"),
        },
    }


def accuracy_evaluation(
    config: PackKVCacheConfig,
    benchmark: str | list[str],
    logger,
    batch_size: int | str = "auto",  # 支持外部传入数字或 "auto"
    limit: int = None,  # 支持 Debug 限制样本数
    output_dir: str = None,  # 支持外部传入输出根路径(如 ./eval_logs)
):
    """在启用KV Cache压缩的情况下, 测量模型在下游任务上的精度损失"""
    logger.info(f"\n{benchmark}: \n{config}")
    PackKVCacheConfigStatic.config = config

    tokenizer = AutoTokenizer.from_pretrained(
        PackKVCacheConfigStatic.config.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_class = MODEL_CLASS_MAP.get(config.model_name)
    if model_class is None:
        raise ValueError(f"Model class not found for {config.model_name}")
    model = model_class.from_pretrained(
        PackKVCacheConfigStatic.config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    model.config.use_cache = True
    torch.set_grad_enabled(False)
    logger.info(f"model class: {model_class.__name__}")
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)
    effective_model_dtype = str(model.dtype)
    task_list = [benchmark] if isinstance(benchmark, str) else benchmark
    with torch.inference_mode():
        results = evaluator.simple_evaluate(
            model=lm_eval_warp,
            tasks=task_list,
            batch_size=batch_size,
            limit=limit,
            log_samples=True,
        )

    if output_dir and len(task_list) == 1:
        current_task = task_list[0]
        model_safe_name = config.model_name.replace("/", "__")
        # 组装动态隔离路径: ./eval_logs/hellaswag/Qwen__Qwen3-8B
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_path = os.path.join(output_dir, current_task, model_safe_name, timestamp)
        os.makedirs(target_path, exist_ok=True)

        # 剥离出 samples 测试细节字典(完全拉齐 lm_eval CLI 保存规则)
        samples_data = results.pop("samples", None)
        # 保存指标结果 results.json
        res_file = os.path.join(target_path, "results.json")
        with open(res_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"已成功将汇总指标保存至: {res_file}")

        # PackKV 配置使用独立 sidecar,保持 lm-eval results.json 标准结构不变.
        packkv_config_file = os.path.join(target_path, "packkv_config.json")
        run_metadata = _accuracy_run_metadata(
            config=config,
            task=current_task,
            batch_size=batch_size,
            limit=limit,
            effective_model_dtype=effective_model_dtype,
        )
        with open(packkv_config_file, "w", encoding="utf-8") as f:
            json.dump(run_metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"已成功保存 PackKV 实验配置至: {packkv_config_file}")

        # 保存细节错题集 samples_task.json
        if samples_data:
            samples_file = os.path.join(target_path, f"samples_{current_task}.json")
            with open(samples_file, "w", encoding="utf-8") as f:
                json.dump(samples_data, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"已成功将测试样本细节(Log Samples)保存至: {samples_file}")
            # 放回字典以防上层调用报错
            results["samples"] = samples_data

    PackKVCacheConfigStatic.config = None
    return results["results"]


def accuracy_evaluation_with_model(
    model,
    tokenizer,
    benchmark: str | list[str],
):
    """直接接收已经实例化好的模型和分词器"""
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    batch_size = 1

    lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)

    results = evaluator.simple_evaluate(
        model=lm_eval_warp,
        tasks=[benchmark] if isinstance(benchmark, str) else benchmark,
        # num_fewshot=0,  # Number of few-shot examples
        batch_size=batch_size,
        # device=_device,
    )

    PackKVCacheConfigStatic.config = None
    return results["results"]


def get_collected_data(logger, model_name="meta-llama/Llama-2-13b-hf"):
    """拦截并收集模型内部Cache数据"""
    PackKVCacheConfigStatic.config = PackKVCacheConfig(
        enable_quant=False,  # 关闭量化
        model_name=model_name,
        quant_method=QuantMethod.PackKV,
        k_block_size=64,
        v_block_size=128,
        k_buffer_size=128,
        v_buffer_size=256,
        k_quant_scale_rel=0.01,
        v_quant_scale_rel=0.01,
    )
    # 收集KV Cache信息
    PackKVCacheConfigStatic.extract_cache = ExtractCacheConfig(collect_round=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_class = MODEL_CLASS_MAP.get(model_name)
    if model_class is None:
        raise ValueError(f"Model class not found for {model_name}")
    model = model_class.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    batch_size = 1
    lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)
    try:
        # 满足收集条件后, 中断评估
        evaluator.simple_evaluate(
            model=lm_eval_warp,
            tasks=["gsm8k"],
            batch_size=batch_size,
        )
    except JumpOutException as e:
        logger.info(e.message)

    rt = PackKVCacheConfigStatic.extract_cache
    PackKVCacheConfigStatic.config = None
    PackKVCacheConfigStatic.extract_cache = None

    return rt


def get_ctx_len_text_from_wikitext_103_v1(ctx_len: int, tokenizer):
    """从真实的维基百科数据集, 提取出精确匹配指定长度的文本, 用于后续的评测"""
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    rt_text = ""
    for i in range(len(dataset["train"])):
        text = dataset["train"][i]["text"]
        rt_text += text
        tokens = tokenizer(rt_text, return_tensors="pt")["input_ids"]
        if tokens.shape[1] > ctx_len:
            break
    return tokenizer(rt_text, return_tensors="pt", truncation=True, max_length=ctx_len)


def save_extract_cache(
    model_name: str, ctx_len: int, extract_cache: ExtractCacheConfig, root_dir: str
):
    """将收集到的KV Cache数据保存到本地磁盘, 以便后续分析和评测使用
    保存数据结构示例:
    {root_dir}/{model_name}/{ctx_len}/{round_i}/[k 或 v]/{layer_idx}.pt
    """
    # check whether the directory exists
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    for round_i in range(extract_cache.collect_round):
        save_dir = os.path.join(root_dir, model_name, str(ctx_len), str(round_i))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        k_dir = os.path.join(save_dir, "k")
        v_dir = os.path.join(save_dir, "v")
        if not os.path.exists(k_dir):
            os.makedirs(k_dir)
        if not os.path.exists(v_dir):
            os.makedirs(v_dir)

        key_caches = PackKVCacheConfigStatic.extract_cache.key_caches[round_i].values()
        value_caches = PackKVCacheConfigStatic.extract_cache.value_caches[
            round_i
        ].values()
        for i, key_cache in enumerate(key_caches):
            torch.save(key_cache.detach().cpu(), os.path.join(k_dir, f"{i}.pt"))
        for i, value_cache in enumerate(value_caches):
            torch.save(value_cache.detach().cpu(), os.path.join(v_dir, f"{i}.pt"))


def load_extract_cache(
    model_name: str, ctx_len: int, root_dir: str, collect_round: int = 1
):
    """
    从本地磁盘读取保存的 KV Cache 数据.
    数据结构对应 save_extract_cache:
    {root_dir}/{model_name}/{ctx_len}/{round_i}/[k 或 v]/{layer_idx}.pt

    Returns:
        成功加载返回组装好的 ExtractCacheConfig 对象;
        如果未命中缓存或文件不完整,返回 None.
    """
    base_dir = os.path.join(root_dir, model_name, str(ctx_len))

    # 1. 检查根目录是否存在,不存在直接未命中
    if not os.path.exists(base_dir):
        return None

    # 2. 实例化一个新的 Cache 容器
    loaded_cache = ExtractCacheConfig(collect_round)

    # 3. 按轮次和层数逐个读取 .pt 文件
    for round_i in range(collect_round):
        round_dir = os.path.join(base_dir, str(round_i))
        k_dir = os.path.join(round_dir, "k")
        v_dir = os.path.join(round_dir, "v")

        # 安全防御:如果目录不完整,说明上次可能没存完就中断了
        if not os.path.exists(k_dir) or not os.path.exists(v_dir):
            print(
                f"[Warning] Cache directory incomplete for round {round_i}. Miss cache."
            )
            return None

        # --- 加载 Key Cache ---
        k_files = [f for f in os.listdir(k_dir) if f.endswith(".pt")]
        if len(k_files) == 0:
            return None  # 文件夹是空的

        # 确保 key_caches 字典里有 round_i 这个键
        if round_i not in loaded_cache.key_caches:
            loaded_cache.key_caches[round_i] = {}

        for k_file in k_files:
            layer_idx = int(k_file.split(".")[0])  # 从 "0.pt" 中提取出层号 0
            file_path = os.path.join(k_dir, k_file)
            # map_location="cpu" 非常重要!防止在纯算力机器上因为 GPU 序号不同而报错
            loaded_cache.key_caches[round_i][layer_idx] = torch.load(
                file_path, map_location="cpu", weights_only=True
            )

        # --- 加载 Value Cache ---
        v_files = [f for f in os.listdir(v_dir) if f.endswith(".pt")]
        if len(v_files) == 0:
            return None

        # 确保 value_caches 字典里有 round_i 这个键
        if round_i not in loaded_cache.value_caches:
            loaded_cache.value_caches[round_i] = {}

        for v_file in v_files:
            layer_idx = int(v_file.split(".")[0])
            file_path = os.path.join(v_dir, v_file)
            loaded_cache.value_caches[round_i][layer_idx] = torch.load(
                file_path, map_location="cpu", weights_only=True
            )

        # 可选防御:检查 k 和 v 读到的层数是否一致
        if len(loaded_cache.key_caches[round_i]) != len(
            loaded_cache.value_caches[round_i]
        ):
            print(f"[Warning] K/V cache layer count mismatch. Miss cache.")
            return None

    # 全部安全加载完成,返回重建的 Cache 对象
    return loaded_cache


def get_pseudo_quant_bit_num(x, min_bits=1):
    """按实际整数范围返回定长存储位宽，不再使用 unique 数量近似."""
    x_for_count = x.detach().to(torch.int64)
    min_value = int(x_for_count.min().item())
    max_value = int(x_for_count.max().item())
    if min_value >= 0:
        return max(min_bits, math.ceil(math.log2(max_value + 1)))

    bits = max(1, min_bits)
    while min_value < -(1 << (bits - 1)) or max_value > (1 << (bits - 1)) - 1:
        bits += 1
    return bits


def get_compressible_prefix_length(seq_len, block_size, recent_size):
    """与在线 Cache 的 cut_tensor 规则一致，只量化完整Block."""
    if block_size <= 0 or recent_size < 0:
        raise ValueError("block_size must be positive and recent_size non-negative")
    if seq_len <= recent_size:
        return 0
    return ((seq_len - recent_size) // block_size) * block_size


def crs_evaluation_with_data(
    config: PackKVCacheConfig,
    key_caches,
    value_caches,
    before_and_after_repacking=None,
    verify_roundtrip: bool = False,
    roundtrip_blocks: int = 1,
    roundtrip_layers: int = 4,
    logger=None,
):
    """按字节统计量化、编码元数据、Recent Buffer和bit-packing开销."""
    metric_names = [
        "original_size",
        "recent_high_precision_size",
        "quant_payload_size",
        "quant_payload_alignment_bits",
        "quant_zero_point_size",
        "quant_scale_size",
        "quant_size",
        "bitpack_payload_size_before_repack",
        "bitpack_min_size_before_repack",
        "bitpack_encode_len_size_before_repack",
        "bitpack_alignment_bits_before_repack",
        "bitpack_code_value_bits_before_repack",
        "bitpack_pack_count_before_repack",
        "bitpack_padding_tokens_before_repack",
        "bitpack_padding_values_before_repack",
        "bit_width_hist_before_repack",
        "encode_size_before_repack",
        "bitpack_payload_size_after_repack",
        "bitpack_min_size_after_repack",
        "bitpack_encode_len_size_after_repack",
        "bitpack_alignment_bits_after_repack",
        "bitpack_code_value_bits_after_repack",
        "bitpack_pack_count_after_repack",
        "bitpack_padding_tokens_after_repack",
        "bitpack_padding_values_after_repack",
        "bit_width_hist_after_repack",
        "permutation_metadata_size",
        "bucket_metadata_size",
        "block_metadata_size",
        "encode_size_after_repack",
        "quant_cr",
        "encode_before_repack_cr",
        "encode_after_repack_cr",
    ]
    res = {
        f"{cache_kind}_{metric}": []
        for cache_kind in ("k", "v")
        for metric in metric_names
    }
    res.update(
        {
            "repack_bucket_count": [],
            "repack_bucket_count_field_bits": [],
            "repack_bucket_metadata_size": [],
            "repack_bucket_occupancy_hist": [],
            "repack_bucket_score_method": [],
            "repack_k_subbucket_count": [],
            "repack_v_subbucket_count": [],
        }
    )
    pa_metric_names = (
        "total_blocks", "total_packs", "candidate_different_packs",
        "payload_beneficial_packs", "positive_delta_candidates",
        "nonpositive_delta_selected_packs",
        "budget_rejected_beneficial_packs", "ceil_selected_packs",
        "ceil_selected_rate", "nearest_nmse_mean", "ceil_nmse_mean",
        "selected_nmse_mean", "nearest_sse", "ceil_sse", "selected_sse",
        "error_budget_sse", "used_delta_sse", "error_budget_utilization",
        "nearest_payload_bits", "ceil_payload_bits",
        "payload_benefit_ceiling_bits", "selected_payload_bits",
        "payload_bits_saved", "error_budget_violations",
    )
    res.update(
        {
            f"{cache_kind}_pa_{metric}": []
            for cache_kind in ("k", "v")
            for metric in pa_metric_names
        }
    )

    layer_num = len(key_caches)
    if layer_num != len(value_caches):
        raise ValueError("K/V cache layer count mismatch")
    if verify_roundtrip:
        if roundtrip_blocks <= 0 or roundtrip_layers <= 0:
            raise ValueError("roundtrip_blocks and roundtrip_layers must be positive")
        audit_layer_count = min(roundtrip_layers, layer_num)
        if audit_layer_count == 1:
            audit_layers = {0}
        else:
            audit_layers = {
                round(index * (layer_num - 1) / (audit_layer_count - 1))
                for index in range(audit_layer_count)
            }
    else:
        audit_layers = set()
    verified_layer_count = 0
    verified_block_count = 0

    for layer_idx in range(layer_num):
        k = key_caches[layer_idx]
        v = value_caches[layer_idx]
        if k.shape[2] != v.shape[2]:
            raise ValueError(f"Layer {layer_idx} K/V sequence length mismatch")

        k_origin_size = k.numel() * k.element_size()
        v_origin_size = v.numel() * v.element_size()
        res["k_original_size"].append(k_origin_size)
        res["v_original_size"].append(v_origin_size)

        compressible_tokens = get_compressible_prefix_length(
            k.shape[2], config.block_size, config.buffer_size
        )
        k_to_quant = k[:, :, :compressible_tokens, :]
        v_to_quant = v[:, :, :compressible_tokens, :]
        k_recent = k[:, :, compressible_tokens:, :]
        v_recent = v[:, :, compressible_tokens:, :]
        k_recent_size = k_recent.numel() * k_recent.element_size()
        v_recent_size = v_recent.numel() * v_recent.element_size()
        res["k_recent_high_precision_size"].append(k_recent_size)
        res["v_recent_high_precision_size"].append(v_recent_size)

        if compressible_tokens == 0:
            for cache_kind, origin_size in (
                ("k", k_origin_size),
                ("v", v_origin_size),
            ):
                for name in (
                    "quant_payload_size",
                    "quant_payload_alignment_bits",
                    "quant_zero_point_size",
                    "quant_scale_size",
                    "bitpack_payload_size_before_repack",
                    "bitpack_min_size_before_repack",
                    "bitpack_encode_len_size_before_repack",
                    "bitpack_alignment_bits_before_repack",
                    "bitpack_code_value_bits_before_repack",
                    "bitpack_pack_count_before_repack",
                    "bitpack_padding_tokens_before_repack",
                    "bitpack_padding_values_before_repack",
                    "bitpack_payload_size_after_repack",
                    "bitpack_min_size_after_repack",
                    "bitpack_encode_len_size_after_repack",
                    "bitpack_alignment_bits_after_repack",
                    "bitpack_code_value_bits_after_repack",
                    "bitpack_pack_count_after_repack",
                    "bitpack_padding_tokens_after_repack",
                    "bitpack_padding_values_after_repack",
                    "permutation_metadata_size",
                    "bucket_metadata_size",
                    "block_metadata_size",
                ):
                    res[f"{cache_kind}_{name}"].append(0)
                res[f"{cache_kind}_bit_width_hist_before_repack"].append("{}")
                res[f"{cache_kind}_bit_width_hist_after_repack"].append("{}")
                res[f"{cache_kind}_quant_size"].append(origin_size)
                res[f"{cache_kind}_encode_size_before_repack"].append(origin_size)
                res[f"{cache_kind}_encode_size_after_repack"].append(origin_size)
                res[f"{cache_kind}_quant_cr"].append(1.0)
                res[f"{cache_kind}_encode_before_repack_cr"].append(1.0)
                res[f"{cache_kind}_encode_after_repack_cr"].append(1.0)
            res["repack_bucket_count"].append(0)
            res["repack_bucket_count_field_bits"].append(0)
            res["repack_bucket_metadata_size"].append(0)
            res["repack_bucket_occupancy_hist"].append("{}")
            res["repack_bucket_score_method"].append("")
            res["repack_k_subbucket_count"].append(0)
            res["repack_v_subbucket_count"].append(0)
            for cache_kind in ("k", "v"):
                for suffix in pa_metric_names:
                    res[f"{cache_kind}_pa_{suffix}"].append(0)
            continue

        fixed_bucket_permutation = None
        fixed_repack_metadata = None
        pa_stats = None
        scale_method = getattr(config, "scale_method", ScaleMethod.CONTINUOUS)
        if scale_method == ScaleMethod.PO2_PACK_AWARE:
            (
                k_quant_int, k_quant_zero, k_quant_scale,
                v_quant_int, v_quant_zero, v_quant_scale,
                fixed_bucket_permutation, fixed_repack_metadata,
                k_pa_stats, v_pa_stats,
            ) = packing_aware_quantize_kv(
                k_to_quant,
                v_to_quant,
                config.block_size,
                config.pack_size,
                config.k_quant_scale_rel,
                config.v_quant_scale_rel,
                config.quant_method.value[0],
                config.quant_method.value[1],
                config.high_precision_zero_point,
                getattr(config, "k_error_budget", 0.1),
                getattr(config, "v_error_budget", 0.1),
                getattr(config, "bucket_count", 4),
                getattr(config, "bucket_score_method", BucketScoreMethod.K_SUM),
            )
            pa_stats = {"k": k_pa_stats, "v": v_pa_stats}
        else:
            k_quant_int, k_quant_zero, k_quant_scale = quant_ints(
                k_to_quant,
                config.block_size,
                config.k_quant_scale_rel,
                config.quant_method.value[0],
                config.high_precision_zero_point,
                scale_method,
            )
            v_quant_int, v_quant_zero, v_quant_scale = quant_ints(
                v_to_quant,
                config.block_size,
                config.v_quant_scale_rel,
                config.quant_method.value[1],
                config.high_precision_zero_point,
                scale_method,
            )
        for cache_kind in ("k", "v"):
            stats = pa_stats[cache_kind] if pa_stats else None
            values = {
                "total_blocks": stats.total_blocks if stats else 0,
                "total_packs": stats.total_packs if stats else 0,
                "candidate_different_packs": stats.candidate_different_packs if stats else 0,
                "payload_beneficial_packs": stats.payload_beneficial_packs if stats else 0,
                "positive_delta_candidates": stats.positive_delta_candidates if stats else 0,
                "nonpositive_delta_selected_packs": stats.nonpositive_delta_selected_packs if stats else 0,
                "budget_rejected_beneficial_packs": stats.budget_rejected_beneficial_packs if stats else 0,
                "ceil_selected_packs": stats.ceil_selected_packs if stats else 0,
                "ceil_selected_rate": stats.ceil_selected_rate if stats else 0,
                "nearest_nmse_mean": stats.nearest_nmse_mean if stats else 0,
                "ceil_nmse_mean": stats.ceil_nmse_mean if stats else 0,
                "selected_nmse_mean": stats.selected_nmse_mean if stats else 0,
                "nearest_sse": stats.nearest_sse if stats else 0,
                "ceil_sse": stats.ceil_sse if stats else 0,
                "selected_sse": stats.selected_sse if stats else 0,
                "error_budget_sse": stats.error_budget_sse if stats else 0,
                "used_delta_sse": stats.used_delta_sse if stats else 0,
                "error_budget_utilization": stats.error_budget_utilization if stats else 0,
                "nearest_payload_bits": stats.nearest_payload_bits if stats else 0,
                "ceil_payload_bits": stats.ceil_payload_bits if stats else 0,
                "payload_benefit_ceiling_bits": stats.payload_benefit_ceiling_bits if stats else 0,
                "selected_payload_bits": stats.selected_payload_bits if stats else 0,
                "payload_bits_saved": stats.payload_bits_saved if stats else 0,
                "error_budget_violations": stats.error_budget_violations if stats else 0,
            }
            for suffix, value in values.items():
                res[f"{cache_kind}_pa_{suffix}"].append(value)
        k_quant_bit_num = get_pseudo_quant_bit_num(k_quant_int)
        v_quant_bit_num = get_pseudo_quant_bit_num(v_quant_int)
        k_quant_payload_bits = k_quant_int.numel() * k_quant_bit_num
        v_quant_payload_bits = v_quant_int.numel() * v_quant_bit_num
        k_quant_payload_size = (k_quant_payload_bits + 7) // 8
        v_quant_payload_size = (v_quant_payload_bits + 7) // 8
        k_zero_size = k_quant_zero.numel() * k_quant_zero.element_size()
        v_zero_size = v_quant_zero.numel() * v_quant_zero.element_size()
        k_scale_size = k_quant_scale.numel() * k_quant_scale.element_size()
        v_scale_size = v_quant_scale.numel() * v_quant_scale.element_size()
        k_quant_size = (
            k_quant_payload_size + k_zero_size + k_scale_size + k_recent_size
        )
        v_quant_size = (
            v_quant_payload_size + v_zero_size + v_scale_size + v_recent_size
        )
        res["k_quant_payload_size"].append(k_quant_payload_size)
        res["v_quant_payload_size"].append(v_quant_payload_size)
        res["k_quant_payload_alignment_bits"].append(
            k_quant_payload_size * 8 - k_quant_payload_bits
        )
        res["v_quant_payload_alignment_bits"].append(
            v_quant_payload_size * 8 - v_quant_payload_bits
        )
        res["k_quant_zero_point_size"].append(k_zero_size)
        res["v_quant_zero_point_size"].append(v_zero_size)
        res["k_quant_scale_size"].append(k_scale_size)
        res["v_quant_scale_size"].append(v_scale_size)
        res["k_quant_size"].append(k_quant_size)
        res["v_quant_size"].append(v_quant_size)
        res["k_quant_cr"].append(k_origin_size / k_quant_size)
        res["v_quant_cr"].append(v_origin_size / v_quant_size)

        if layer_idx in audit_layers:
            audit = verify_repack_bitstream_roundtrip(
                k_quant_int,
                v_quant_int,
                config.pack_size,
                config.repack_method,
                max_blocks=roundtrip_blocks,
                bucket_count=getattr(config, "bucket_count", 4),
                bucket_score_method=getattr(
                    config,
                    "bucket_score_method",
                    BucketScoreMethod.K_SUM,
                ),
            )
            verified_layer_count += 1
            verified_block_count += audit.verified_blocks
            if logger is not None:
                logger.info(
                    "Round-trip PASS: layer=%d blocks=%d K=%dB V=%dB bucket=%dB total=%dB",
                    layer_idx,
                    audit.verified_blocks,
                    audit.k_stream_bytes,
                    audit.v_stream_bytes,
                    audit.bucket_metadata_bytes,
                    audit.total_bytes,
                )

        if config.quant_method == QuantMethod.PackKV:
            (
                k_stats_before,
                v_stats_before,
                k_stats_after,
                v_stats_after,
                repack_metadata,
            ) = repack_and_encode(
                k_quant_int,
                v_quant_int,
                config.pack_size,
                config.repack_method,
                before_and_after_repacking,
                return_stats=True,
                bucket_count=getattr(config, "bucket_count", 4),
                bucket_score_method=getattr(
                    config,
                    "bucket_score_method",
                    BucketScoreMethod.COMBINED_SUM,
                ),
                return_repack_metadata=True,
                fixed_bucket_permutation=fixed_bucket_permutation,
                fixed_repack_metadata=fixed_repack_metadata,
            )
            shared_bucket_metadata_size = repack_metadata.bucket_metadata_bytes
            res["repack_bucket_count"].append(repack_metadata.bucket_count)
            res["repack_bucket_count_field_bits"].append(
                repack_metadata.bucket_count_field_bits
            )
            res["repack_bucket_metadata_size"].append(
                shared_bucket_metadata_size
            )
            res["repack_bucket_occupancy_hist"].append(
                json.dumps(
                    repack_metadata.bucket_occupancy_histogram or {},
                    sort_keys=True,
                )
            )
            res["repack_bucket_score_method"].append(
                repack_metadata.bucket_score_method
            )
            res["repack_k_subbucket_count"].append(
                repack_metadata.k_subbucket_count
            )
            res["repack_v_subbucket_count"].append(
                repack_metadata.v_subbucket_count
            )

            for cache_kind, stats_before, stats_after, origin_size, zero_size, scale_size, recent_size in (
                (
                    "k",
                    k_stats_before,
                    k_stats_after,
                    k_origin_size,
                    k_zero_size,
                    k_scale_size,
                    k_recent_size,
                ),
                (
                    "v",
                    v_stats_before,
                    v_stats_after,
                    v_origin_size,
                    v_zero_size,
                    v_scale_size,
                    v_recent_size,
                ),
            ):
                for phase, stats in (
                    ("before_repack", stats_before),
                    ("after_repack", stats_after),
                ):
                    res[f"{cache_kind}_bitpack_payload_size_{phase}"].append(
                        stats.payload_bytes
                    )
                    res[f"{cache_kind}_bitpack_min_size_{phase}"].append(
                        stats.pack_min_bytes
                    )
                    res[f"{cache_kind}_bitpack_encode_len_size_{phase}"].append(
                        stats.encode_length_bytes
                    )
                    res[f"{cache_kind}_bitpack_alignment_bits_{phase}"].append(
                        stats.byte_alignment_bits
                    )
                    res[f"{cache_kind}_bitpack_code_value_bits_{phase}"].append(
                        stats.code_value_bits
                    )
                    res[f"{cache_kind}_bitpack_pack_count_{phase}"].append(
                        stats.pack_count
                    )
                    res[f"{cache_kind}_bitpack_padding_tokens_{phase}"].append(
                        stats.padded_token_count
                    )
                    res[f"{cache_kind}_bitpack_padding_values_{phase}"].append(
                        stats.padded_value_count
                    )
                    res[f"{cache_kind}_bit_width_hist_{phase}"].append(
                        json.dumps(stats.bit_width_histogram, sort_keys=True)
                    )

                # K/V 使用同一物理重排;共享的 bucket count 元数据在两者之间
                # 均分,从而保证 K+V 总存储只计一次.
                permutation_metadata_size = 0
                if cache_kind == "k":
                    bucket_metadata_size = (
                        shared_bucket_metadata_size + 1
                    ) // 2
                else:
                    bucket_metadata_size = shared_bucket_metadata_size // 2
                block_metadata_size = 0
                res[f"{cache_kind}_permutation_metadata_size"].append(
                    permutation_metadata_size
                )
                res[f"{cache_kind}_bucket_metadata_size"].append(
                    bucket_metadata_size
                )
                res[f"{cache_kind}_block_metadata_size"].append(
                    block_metadata_size
                )
                encoded_before = (
                    stats_before.total_bytes
                    + zero_size
                    + scale_size
                    + recent_size
                )
                encoded_after = (
                    stats_after.total_bytes
                    + zero_size
                    + scale_size
                    + recent_size
                    + permutation_metadata_size
                    + bucket_metadata_size
                    + block_metadata_size
                )
                res[f"{cache_kind}_encode_size_before_repack"].append(
                    encoded_before
                )
                res[f"{cache_kind}_encode_size_after_repack"].append(encoded_after)
                res[f"{cache_kind}_encode_before_repack_cr"].append(
                    origin_size / encoded_before
                )
                res[f"{cache_kind}_encode_after_repack_cr"].append(
                    origin_size / encoded_after
                )
    if verify_roundtrip and logger is not None:
        logger.info(
            "Round-trip audit PASS: verified_layers=%d/%d verified_blocks=%d",
            verified_layer_count,
            layer_num,
            verified_block_count,
        )
    return res


def crs_evaluation_with_data_detail_rebuttal(
    config: PackKVCacheConfig, key_caches, value_caches, before_and_after_repacking=None
):
    """为硬件开销分析量身定制的, 将数据载荷和元数据的大小分开计算"""
    # assert key_caches[0].shape[1] == 32 and key_caches[0].shape[3] == 128
    res = {
        "k_original_size": [],
        "v_original_size": [],
        "k_quant_zero_point_size": [],
        "v_quant_zero_point_size": [],
        "k_quant_scale_size": [],
        "v_quant_scale_size": [],
        "k_recent_high_precision_size": [],
        "v_recent_high_precision_size": [],
        "k_bitpack_min_value_size": [],
        "v_bitpack_min_value_size": [],
        "k_bitpack_encode_len_size": [],
        "v_bitpack_encode_len_size": [],
        "k_bitpack_encoded_size": [],
        "v_bitpack_encoded_size": [],
        "k_bitpack_alignment_bits": [],
        "v_bitpack_alignment_bits": [],
    }

    layer_num = len(key_caches)

    for layer_idx in range(layer_num):
        k = key_caches[layer_idx]
        v = value_caches[layer_idx]
        k_origin_size = k.numel() * k.element_size()
        v_origin_size = v.numel() * v.element_size()
        res["k_original_size"].append(k_origin_size)
        res["v_original_size"].append(v_origin_size)
        compressible_tokens = get_compressible_prefix_length(
            k.shape[2], config.block_size, config.buffer_size
        )
        k_to_quant = k[:, :, :compressible_tokens, :]
        v_to_quant = v[:, :, :compressible_tokens, :]
        k_recent = k[:, :, compressible_tokens:, :]
        v_recent = v[:, :, compressible_tokens:, :]
        res["k_recent_high_precision_size"].append(
            k_recent.numel() * k_recent.element_size()
        )
        res["v_recent_high_precision_size"].append(
            v_recent.numel() * v_recent.element_size()
        )
        if compressible_tokens == 0:
            for cache_kind in ("k", "v"):
                for name in (
                    "quant_zero_point_size",
                    "quant_scale_size",
                    "bitpack_min_value_size",
                    "bitpack_encode_len_size",
                    "bitpack_encoded_size",
                    "bitpack_alignment_bits",
                ):
                    res[f"{cache_kind}_{name}"].append(0)
            continue

        k_quant_int, k_quant_zero, k_quant_scale = quant_ints(
            k_to_quant,
            config.block_size,
            config.k_quant_scale_rel,
            config.quant_method.value[0],
            config.high_precision_zero_point,
            getattr(config, "scale_method", ScaleMethod.CONTINUOUS),
        )
        v_quant_int, v_quant_zero, v_quant_scale = quant_ints(
            v_to_quant,
            config.block_size,
            config.v_quant_scale_rel,
            config.quant_method.value[1],
            config.high_precision_zero_point,
            getattr(config, "scale_method", ScaleMethod.CONTINUOUS),
        )
        res["k_quant_zero_point_size"].append(
            k_quant_zero.numel() * k_quant_zero.element_size()
        )
        res["v_quant_zero_point_size"].append(
            v_quant_zero.numel() * v_quant_zero.element_size()
        )
        res["k_quant_scale_size"].append(
            k_quant_scale.numel() * k_quant_scale.element_size()
        )
        res["v_quant_scale_size"].append(
            v_quant_scale.numel() * v_quant_scale.element_size()
        )
        if config.quant_method == QuantMethod.PackKV:
            (
                _,
                _,
                k_stats,
                v_stats,
            ) = repack_and_encode(
                k_quant_int,
                v_quant_int,
                config.pack_size,
                config.repack_method,
                return_stats=True,
                bucket_count=getattr(config, "bucket_count", 4),
                bucket_score_method=getattr(
                    config,
                    "bucket_score_method",
                    BucketScoreMethod.COMBINED_SUM,
                ),
            )
            for cache_kind, stats in (("k", k_stats), ("v", v_stats)):
                res[f"{cache_kind}_bitpack_min_value_size"].append(
                    stats.pack_min_bytes
                )
                res[f"{cache_kind}_bitpack_encode_len_size"].append(
                    stats.encode_length_bytes
                )
                res[f"{cache_kind}_bitpack_encoded_size"].append(
                    stats.payload_bytes
                )
                res[f"{cache_kind}_bitpack_alignment_bits"].append(
                    stats.byte_alignment_bits
                )
    return res


def repacking_throughput_with_kv_rebuttal(
    config: PackKVCacheConfig,
    key_caches,
    value_caches,
):
    """计算算法本身的延迟"""
    # assert key_caches[0].shape[1] == 32 and key_caches[0].shape[3] == 128
    res = {"greedy_repacking_time": [], "median_repacking_time": []}

    layer_num = len(key_caches)

    for layer_idx in range(layer_num):
        k = key_caches[layer_idx]
        v = value_caches[layer_idx]
        k_quant_int, k_quant_zero, k_quant_scale = quant_ints(
            k,
            config.block_size,
            config.k_quant_scale_rel,
            config.quant_method.value[0],
            config.high_precision_zero_point,
            getattr(config, "scale_method", ScaleMethod.CONTINUOUS),
        )
        v_quant_int, v_quant_zero, v_quant_scale = quant_ints(
            v,
            config.block_size,
            config.v_quant_scale_rel,
            config.quant_method.value[1],
            config.high_precision_zero_point,
            getattr(config, "scale_method", ScaleMethod.CONTINUOUS),
        )
        if config.quant_method == QuantMethod.PackKV:
            greedy_time, median_time = repack_throughput_detail_rebuttal(
                k_quant_int, v_quant_int, config.pack_size
            )
            res["greedy_repacking_time"].append(greedy_time)
            res["median_repacking_time"].append(median_time)

    return res


def cr_evaluation(
    config: PackKVCacheConfig,
    ctx_len: int,
    enable_save: bool,
    logger,
    collect_round: int = 1,
    before_and_after_repacking=None,
    verify_roundtrip: bool = False,
    roundtrip_blocks: int = 1,
    roundtrip_layers: int = 4,
):
    logger.info(f"ctx_len: {ctx_len}")
    logger.info(config)
    # logger.info(f"Created a temp config for cache collection and set its enable_quant to False.")

    # 关闭量化, 准备提取高精度的KV Cache
    PackKVCacheConfigStatic.config = PackKVCacheConfig(
        enable_quant=False,
        model_name=config.model_name,
        quant_method=config.quant_method,
        repack_method=config.repack_method,
        high_precision_zero_point=config.high_precision_zero_point,
        block_size=config.block_size,
        buffer_size=config.buffer_size,
        pack_size=config.pack_size,
        k_quant_scale_rel=config.k_quant_scale_rel,
        v_quant_scale_rel=config.v_quant_scale_rel,
    )
    # KV Cache缓存路径
    cache_dir = "./dumped_cache"
    logger.info("Checking for existing KV cache on disk...")
    loaded_cache = load_extract_cache(
        config.model_name, ctx_len, "./dumped_cache", collect_round
    )
    if loaded_cache is not None:
        # [命中缓存]
        logger.info("Cache HIT! Successfully loaded high-precision KV Cache from disk.")
        PackKVCacheConfigStatic.extract_cache = loaded_cache
    else:
        logger.warning(
            "Cache MISS. Loading LLM weights to extract KV Cache. This may take a while..."
        )
        PackKVCacheConfigStatic.extract_cache = ExtractCacheConfig(collect_round)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model_class = MODEL_CLASS_MAP.get(config.model_name)

        if model_class is None:
            raise ValueError(f"Model class not found for {config.model_name}")
        model = model_class.from_pretrained(
            PackKVCacheConfigStatic.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            # attn_implementation="sdpa",
            # attn_implementation="eager",
            trust_remote_code=True,
        )
        model.eval()
        # batch_size = 1
        # lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)
        inputs = get_ctx_len_text_from_wikitext_103_v1(ctx_len, tokenizer).to(
            model.device
        )

        with torch.no_grad():
            _ = model(
                **inputs,
                use_cache=True,
                logits_to_keep=1,
            )
        PackKVCachePytorchQuant.round_ = 0

        if enable_save:
            save_extract_cache(
                config.model_name,
                ctx_len,
                PackKVCacheConfigStatic.extract_cache,
                "./dumped_cache",
            )

        # 释放显存
        del model
        torch.cuda.empty_cache()

    rts = []
    logger.info(
        f"Cache Size {PackKVCacheConfigStatic.extract_cache.size() / 8 / 1024 / 1024 / 1024} GB"
    )

    for round_i in range(collect_round):
        key_caches = list(
            PackKVCacheConfigStatic.extract_cache.key_caches[round_i].values()
        )
        value_caches = list(
            PackKVCacheConfigStatic.extract_cache.value_caches[round_i].values()
        )
        rts.append(
            crs_evaluation_with_data(
                config,
                key_caches,
                value_caches,
                before_and_after_repacking,
                verify_roundtrip=verify_roundtrip,
                roundtrip_blocks=roundtrip_blocks,
                roundtrip_layers=roundtrip_layers,
                logger=logger,
            )
        )

    PackKVCacheConfigStatic.config = None
    PackKVCacheConfigStatic.extract_cache = None
    return rts


def k_cpu_compress_gpu_mat_vec_mul(
    k: torch.Tensor, config
) -> Tuple[int, float, float, float]:
    """在CPU上进行位打包压缩, 同时测量压缩过程和GPU上矩阵向量乘的时间"""
    if config.high_precision_zero_point:
        raise ValueError("当前 fused KQ kernel 只支持整数 zero point")
    k_quant_int, k_quant_zero, k_quant_scale = quant_ints(
        k,
        config.block_size,
        config.k_quant_scale_rel,
        config.quant_method.value[0],
        config.high_precision_zero_point,
        getattr(config, "scale_method", ScaleMethod.CONTINUOUS),
    )
    dtype_ = k.dtype
    k = dequantize_ints(
        k_quant_int,
        k_quant_zero,
        k_quant_scale,
        config.high_precision_zero_point,
    ).flatten(2, 3)
    k_quant_int = (
        k_quant_int.flatten(2, 3).permute(2, 0, 1, 3).flatten(1, 3).contiguous()
    )
    k_quant_zero = k_quant_zero.flatten()
    k_quant_scale = k_quant_scale.flatten()

    ctx_len_block_size = 64
    hidden_dim_block_size = 128

    ctx_len, hidden_dim = k_quant_int.shape
    ctx_len_block_num = ctx_len // ctx_len_block_size
    hidden_dim_block_num = hidden_dim // hidden_dim_block_size
    k_bit_len = math.ceil(math.log2(k_quant_int.unique().numel()))
    assert k_bit_len == 4

    block = k_quant_int.to(torch.uint8).cpu()
    block_info_buffer = torch.zeros(
        ctx_len_block_num * hidden_dim_block_num, 2, dtype=torch.uint32
    ).cpu()

    compressed_buffer = torch.zeros_like(block)

    compressed_size = k_encode_cpu(
        block,
        compressed_buffer,
        block_info_buffer,
        ctx_len,
        hidden_dim,
        ctx_len_block_size,
        hidden_dim_block_size,
        k_bit_len,
    )

    # compressed_buffer = compressed_buffer.flatten()[:compressed_size].clone()

    decompress_tensor = torch.zeros_like(block)

    k_decode_cpu(
        compressed_buffer,
        block_info_buffer,
        decompress_tensor,
        ctx_len,
        hidden_dim,
        ctx_len_block_size,
        hidden_dim_block_size,
        k_bit_len,
    )
    # 保证压缩, 解压前后一致
    assert (block == decompress_tensor).all()

    head_num = hidden_dim // hidden_dim_block_size
    q = torch.randn(head_num, hidden_dim_block_size, dtype=dtype_).to(k.device)
    # 因为k被高度压缩和重新排布了,提前把Q张量的内存排布也打乱了,
    # 使之与K的压缩块在物理内存上一一对应.
    q_reshaped = q.view(head_num, 128 // 4, 4)
    part1 = q_reshaped[:, :, :2].contiguous().view(head_num, -1)
    part2 = q_reshaped[:, :, 2:].contiguous().view(head_num, -1)
    q_transform = torch.cat([part1, part2], dim=1)

    t = torch.zeros(head_num, ctx_len, dtype=dtype_).to(k.device)

    our_kernel_time = kq_mat_vec_mul(
        compressed_buffer.to(k.device),
        block_info_buffer.to(k.device),
        q_transform,
        t,
        ctx_len,
        hidden_dim,
        ctx_len_block_size,
        hidden_dim_block_size,
        k_bit_len,
    )

    # print("q.shape: ", q.shape, "dtype: ", q.dtype, "is_contiguous: ", q.is_contiguous())
    # print("k_quant_zero.shape", k_quant_zero.shape, "dtype: ", k_quant_zero.dtype, "is_contiguous: ", k_quant_zero.is_contiguous())
    # print("k_quant_scale.shape", k_quant_scale.shape, "dtype: ", k_quant_scale.dtype, "is_contiguous: ", k_quant_scale.is_contiguous())
    # print("t.shape", t.shape, "dtype: ", t.dtype, "is_contiguous: ", t.is_contiguous())

    # Replace Pytorch implementation with our fused CUDA kernel
    our_kq = torch.empty_like(t)
    # 在 GPU 寄存器里瞬间解压并算乘法
    fused_kernel_time = fused_kq(our_kq, q, k_quant_zero, k_quant_scale, t)
    our_none_kernel_time = fused_kernel_time

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    k_ = k.squeeze(0)
    q_ = q.unsqueeze(2)
    torch.cuda.synchronize()
    # 跑PyTorch原生算子, 作为裁判基准线
    start.record()
    target_kq = k_ @ q_
    end.record()
    torch.cuda.synchronize()
    pytorch_kernel_time = start.elapsed_time(end)

    target_kq = target_kq.squeeze()

    # 算子精度验收
    max_error_percentage = (
        (target_kq - our_kq).abs() / target_kq.abs().max()
    ).max() * 100

    # print(f"max_error_percentage: {max_error_percentage}")
    if max_error_percentage > 10:
        print(f"max_error_percentage: {max_error_percentage}")
        exit()

    # print(f"our_kernel_time: {our_kernel_time}, our_none_kernel_time: {our_none_kernel_time}, pytorch_kernel_time: {pytorch_kernel_time}")
    # exit()

    return compressed_size, our_kernel_time, our_none_kernel_time, pytorch_kernel_time


def k_cpu_compress(k: torch.Tensor, config) -> Tuple[int, float, float, float]:
    """把原始的高精度 Cache 张量送到 CPU 上进行位打包压缩,
    并精确记录压缩所需的时间和最终大小"""
    compress_size = k.numel() * k.element_size()
    compress_time = 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    k_quant_int, k_quant_zero, k_quant_scale = quant_ints(
        k,
        config.block_size,
        config.k_quant_scale_rel,
        config.quant_method.value[0],
        config.high_precision_zero_point,
        getattr(config, "scale_method", ScaleMethod.CONTINUOUS),
    )
    dtype_ = k.dtype
    end.record()
    torch.cuda.synchronize()
    compress_time += start.elapsed_time(end)
    # 量化与显存排布转换
    k = dequantize_ints(
        k_quant_int,
        k_quant_zero,
        k_quant_scale,
        config.high_precision_zero_point,
    ).flatten(2, 3)
    k_quant_int = (
        k_quant_int.flatten(2, 3).permute(2, 0, 1, 3).flatten(1, 3).contiguous()
    )
    k_quant_zero = k_quant_zero.flatten()
    k_quant_scale = k_quant_scale.flatten()

    ctx_len_block_size = 64
    hidden_dim_block_size = 128

    ctx_len, hidden_dim = k_quant_int.shape
    ctx_len_block_num = ctx_len // ctx_len_block_size
    hidden_dim_block_num = hidden_dim // hidden_dim_block_size
    k_bit_len = math.ceil(math.log2(k_quant_int.unique().numel()))
    assert k_bit_len == 4

    block = k_quant_int.to(torch.uint8).cpu()
    block_info_buffer = torch.zeros(
        ctx_len_block_num * hidden_dim_block_num, 2, dtype=torch.uint32
    ).cpu()

    compressed_buffer = torch.zeros_like(block)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    # 调用位移位打包的CPU实现
    compressed_size = k_encode_cpu(
        block,
        compressed_buffer,
        block_info_buffer,
        ctx_len,
        hidden_dim,
        ctx_len_block_size,
        hidden_dim_block_size,
        k_bit_len,
    )
    end.record()
    torch.cuda.synchronize()
    compress_time += start.elapsed_time(end)

    return compress_size, compress_time


def v_cpu_compress_gpu_mat_vec_mul(
    v: torch.Tensor, config
) -> Tuple[int, float, float, float]:
    # return 0,0,0,0
    if config.high_precision_zero_point:
        raise ValueError("当前 fused WV kernel 只支持整数 zero point")
    v_quant_int, v_quant_zero, v_quant_scale = quant_ints(
        v,
        config.block_size,
        config.v_quant_scale_rel,
        config.quant_method.value[1],
        config.high_precision_zero_point,
        getattr(config, "scale_method", ScaleMethod.CONTINUOUS),
    )
    dtype_ = v.dtype
    v = dequantize_ints(
        v_quant_int,
        v_quant_zero,
        v_quant_scale,
        config.high_precision_zero_point,
    ).flatten(2, 3)
    v_quant_int = (
        v_quant_int.flatten(2, 3).permute(2, 0, 1, 3).flatten(1, 3).contiguous()
    )
    v_quant_zero = v_quant_zero.flatten()
    v_quant_scale = v_quant_scale.flatten()

    ctx_len_block_size = 128
    hidden_dim_block_size = 64
    head_dim = 128

    ctx_len, hidden_dim = v_quant_int.shape
    ctx_len_block_num = ctx_len // ctx_len_block_size
    hidden_dim_block_num = hidden_dim // hidden_dim_block_size
    v_bit_len = math.ceil(math.log2(v_quant_int.unique().numel()))

    block = v_quant_int.to(torch.uint8).cpu()
    block_info_buffer = torch.zeros(
        ctx_len_block_num * hidden_dim_block_num, 2, dtype=torch.uint32
    ).cpu()

    compressed_buffer = torch.zeros_like(block)

    compressed_size = v_encode_cpu(
        block,
        compressed_buffer,
        block_info_buffer,
        ctx_len,
        hidden_dim,
        ctx_len_block_size,
        hidden_dim_block_size,
        v_bit_len,
    )

    decompress_tensor = torch.zeros_like(block)

    v_decode_cpu(
        compressed_buffer,
        block_info_buffer,
        decompress_tensor,
        ctx_len,
        hidden_dim,
        ctx_len_block_size,
        hidden_dim_block_size,
        v_bit_len,
    )

    assert (block == decompress_tensor).all()

    head_num = math.ceil(hidden_dim / head_dim)
    w = torch.randn(head_num, ctx_len, dtype=dtype_).to(v.device).pow(2).softmax(dim=1)

    term1 = torch.zeros(
        # ctx_len_block_num,
        # 1,
        head_num,
        head_dim,
        dtype=torch.float32,
    ).to(v.device)

    w_prime = w * v_quant_scale.unsqueeze(0)

    v_our_kernel_time = wv_mat_vec_mul(
        compressed_buffer.to(v.device),
        block_info_buffer.to(v.device),
        w_prime,
        term1,
        ctx_len,
        hidden_dim,
        ctx_len_block_size,
        hidden_dim_block_size,
        v_bit_len,
    )

    term1 = term1.to(dtype_)

    # print(f"term1.shape: {term1.shape}, dtype: {term1.dtype}, is_contiguous: {term1.is_contiguous()}")
    # print(f"v_quant_zero.shape: {v_quant_zero.shape}, dtype: {v_quant_zero.dtype}, is_contiguous: {v_quant_zero.is_contiguous()}")
    # print(f"v_quant_scale.shape: {v_quant_scale.shape}, dtype: {v_quant_scale.dtype}, is_contiguous: {v_quant_scale.is_contiguous()}")
    # print(f"w.shape: {w.shape}, dtype: {w.dtype}, is_contiguous: {w.is_contiguous()}")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # term1 = term1.unsqueeze(0)

    our_wv = torch.empty_like(term1)
    v_our_none_kernel_time = fused_wv(our_wv, w, v_quant_zero, v_quant_scale, term1)

    w_ = w.unsqueeze(1)
    v_ = v.squeeze()

    torch.cuda.synchronize()
    start.record()
    target_wv = w_ @ v_
    end.record()
    torch.cuda.synchronize()
    v_pytorch_kernel_time = start.elapsed_time(end)

    target_wv = target_wv.squeeze()

    max_error_percentage = (
        (target_wv - our_wv).abs() / target_wv.abs().max()
    ).max() * 100

    # print(f"max_error_percentage: {max_error_percentage}")
    if max_error_percentage > 6:
        print(f"max_error_percentage: {max_error_percentage}")
        exit()

    # print(f"v_our_kernel_time: {v_our_kernel_time}, v_our_none_kernel_time: {v_our_none_kernel_time}, v_pytorch_kernel_time: {v_pytorch_kernel_time}")

    # exit()

    return (
        compressed_size,
        v_our_kernel_time,
        v_our_none_kernel_time,
        v_pytorch_kernel_time,
    )


def v_cpu_compress(v: torch.Tensor, config) -> Tuple[int, float, float, float]:
    compress_size = v.numel() * v.element_size()
    compress_time = 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    v_quant_int, v_quant_zero, v_quant_scale = quant_ints(
        v,
        config.block_size,
        config.v_quant_scale_rel,
        config.quant_method.value[1],
        config.high_precision_zero_point,
        getattr(config, "scale_method", ScaleMethod.CONTINUOUS),
    )
    end.record()
    torch.cuda.synchronize()
    compress_time += start.elapsed_time(end)
    dtype_ = v.dtype
    v = dequantize_ints(
        v_quant_int,
        v_quant_zero,
        v_quant_scale,
        config.high_precision_zero_point,
    ).flatten(2, 3)
    v_quant_int = (
        v_quant_int.flatten(2, 3).permute(2, 0, 1, 3).flatten(1, 3).contiguous()
    )
    v_quant_zero = v_quant_zero.flatten()
    v_quant_scale = v_quant_scale.flatten()

    ctx_len_block_size = 128
    hidden_dim_block_size = 64
    head_dim = 128

    ctx_len, hidden_dim = v_quant_int.shape
    ctx_len_block_num = ctx_len // ctx_len_block_size
    hidden_dim_block_num = hidden_dim // hidden_dim_block_size
    v_bit_len = math.ceil(math.log2(v_quant_int.unique().numel()))

    block = v_quant_int.to(torch.uint8).cpu()
    block_info_buffer = torch.zeros(
        ctx_len_block_num * hidden_dim_block_num, 2, dtype=torch.uint32
    ).cpu()

    compressed_buffer = torch.zeros_like(block)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    compressed_size = v_encode_cpu(
        block,
        compressed_buffer,
        block_info_buffer,
        ctx_len,
        hidden_dim,
        ctx_len_block_size,
        hidden_dim_block_size,
        v_bit_len,
    )
    end.record()
    torch.cuda.synchronize()
    compress_time += start.elapsed_time(end)

    return compress_size, compress_time


def throughputs_evaluation_with_data(
    config: PackKVCacheConfig, key_caches, value_caches
):
    # assert key_caches[0].shape[1] == 32 and key_caches[0].shape[3] == 128
    res = {
        "k_original_size": [],
        "v_original_size": [],
        "k_compressed_size": [],
        "v_compressed_size": [],
        "k_our_kernel_time": [],
        "k_our_none_kernel_time": [],
        "k_pytorch_kernel_time": [],
        "v_our_kernel_time": [],
        "v_our_none_kernel_time": [],
        "v_pytorch_kernel_time": [],
        "k_compress_size": 0,
        "v_compress_size": 0,
        "k_compress_time": 0.0,
        "v_compress_time": 0.0,
    }

    layer_num = len(key_caches)

    # k warm up
    for layer_idx in range(3):
        k = key_caches[layer_idx]
        k_cpu_compress_gpu_mat_vec_mul(k, config)

    for layer_idx in range(layer_num):
        k = key_caches[layer_idx]
        # assert k.dtype == torch.float16, "kernel only support fp16 for now"
        k_origin_size = k.numel() * k.element_size()
        res["k_original_size"].append(k_origin_size)
        # k_transform = k.permute(2, 0, 1, 3).flatten(1,3).contiguous()
        # v_transform = v.permute(2, 0, 1, 3).flatten(1,3).contiguous()
        (
            k_compress_size,
            k_our_kernel_time,
            k_our_none_kernel_time,
            k_pytorch_kernel_time,
        ) = k_cpu_compress_gpu_mat_vec_mul(k, config)
        res["k_compressed_size"].append(k_compress_size)
        res["k_our_kernel_time"].append(k_our_kernel_time)
        res["k_our_none_kernel_time"].append(k_our_none_kernel_time)
        res["k_pytorch_kernel_time"].append(k_pytorch_kernel_time)
        # res["k_compress_size"].append(k_compress_size)

    # v warm up
    for layer_idx in range(3):
        v = value_caches[layer_idx]

        v_cpu_compress_gpu_mat_vec_mul(v, config)

    for layer_idx in range(layer_num):
        v = value_caches[layer_idx]
        # assert v.dtype == torch.float16, "kernel only support fp16 for now"
        v_origin_size = v.numel() * v.element_size()
        res["v_original_size"].append(v_origin_size)
        # k_transform = k.permute(2, 0, 1, 3).flatten(1,3).contiguous()
        # v_transform = v.permute(2, 0, 1, 3).flatten(1,3).contiguous()
        (
            v_compress_size,
            v_our_kernel_time,
            v_our_none_kernel_time,
            v_pytorch_kernel_time,
        ) = v_cpu_compress_gpu_mat_vec_mul(v, config)
        res["v_compressed_size"].append(v_compress_size)
        res["v_our_kernel_time"].append(v_our_kernel_time)
        res["v_our_none_kernel_time"].append(v_our_none_kernel_time)
        res["v_pytorch_kernel_time"].append(v_pytorch_kernel_time)
        # res["k_compress_size"].append(k_compress_size)

    return res


def throughputs_evaluation_with_data_rebuttal(
    config: PackKVCacheConfig, key_caches, value_caches, ctx_len_factor
):
    # assert key_caches[0].shape[1] == 32 and key_caches[0].shape[3] == 128
    res = {
        "k_original_size": [],
        "v_original_size": [],
        "k_compressed_size": [],
        "v_compressed_size": [],
        "k_our_kernel_time": [],
        "k_our_none_kernel_time": [],
        "k_pytorch_kernel_time": [],
        "v_our_kernel_time": [],
        "v_our_none_kernel_time": [],
        "v_pytorch_kernel_time": [],
        "k_compress_size": 0,
        "v_compress_size": 0,
        "k_compress_time": 0.0,
        "v_compress_time": 0.0,
    }

    for index in range(len(key_caches)):
        key_caches[index] = torch.concat([key_caches[index]] * ctx_len_factor, dim=2)
        value_caches[index] = torch.concat(
            [value_caches[index]] * ctx_len_factor, dim=2
        )

    layer_num = len(key_caches)

    # k warm up
    for layer_idx in range(3):
        k = key_caches[layer_idx]
        k_cpu_compress_gpu_mat_vec_mul(k, config)

    for layer_idx in range(layer_num):
        k = key_caches[layer_idx]
        # assert k.dtype == torch.float16, "kernel only support fp16 for now"
        k_origin_size = k.numel() * k.element_size()
        res["k_original_size"].append(k_origin_size)
        # k_transform = k.permute(2, 0, 1, 3).flatten(1,3).contiguous()
        # v_transform = v.permute(2, 0, 1, 3).flatten(1,3).contiguous()
        (
            k_compress_size,
            k_our_kernel_time,
            k_our_none_kernel_time,
            k_pytorch_kernel_time,
        ) = k_cpu_compress_gpu_mat_vec_mul(k, config)
        res["k_compressed_size"].append(k_compress_size)
        res["k_our_kernel_time"].append(k_our_kernel_time)
        res["k_our_none_kernel_time"].append(k_our_none_kernel_time)
        res["k_pytorch_kernel_time"].append(k_pytorch_kernel_time)
        # res["k_compress_size"].append(k_compress_size)

    # v warm up
    for layer_idx in range(3):
        v = value_caches[layer_idx]

        v_cpu_compress_gpu_mat_vec_mul(v, config)

    for layer_idx in range(layer_num):
        v = value_caches[layer_idx]
        # assert v.dtype == torch.float16, "kernel only support fp16 for now"
        v_origin_size = v.numel() * v.element_size()
        res["v_original_size"].append(v_origin_size)
        # k_transform = k.permute(2, 0, 1, 3).flatten(1,3).contiguous()
        # v_transform = v.permute(2, 0, 1, 3).flatten(1,3).contiguous()
        (
            v_compress_size,
            v_our_kernel_time,
            v_our_none_kernel_time,
            v_pytorch_kernel_time,
        ) = v_cpu_compress_gpu_mat_vec_mul(v, config)
        res["v_compressed_size"].append(v_compress_size)
        res["v_our_kernel_time"].append(v_our_kernel_time)
        res["v_our_none_kernel_time"].append(v_our_none_kernel_time)
        res["v_pytorch_kernel_time"].append(v_pytorch_kernel_time)
        # res["k_compress_size"].append(k_compress_size)

    return res


def compressor_throughputs_evaluation_with_data(
    config: PackKVCacheConfig, key_caches, value_caches
):
    # assert key_caches[0].shape[1] == 32 and key_caches[0].shape[3] == 128
    res = {
        "k_compress_size": [],
        "v_compress_size": [],
        "k_compress_time": [],
        "v_compress_time": [],
    }

    layer_num = len(key_caches)

    # k warm up
    for layer_idx in range(3):
        k = key_caches[layer_idx]
        k_cpu_compress(k, config)

    for layer_idx in range(layer_num):
        k = key_caches[layer_idx]
        k_compress_size, k_compress_time = k_cpu_compress(k, config)
        res["k_compress_size"].append(k_compress_size)
        res["k_compress_time"].append(k_compress_time)

    # v warm up
    for layer_idx in range(3):
        v = value_caches[layer_idx]

        v_cpu_compress(v, config)

    for layer_idx in range(layer_num):
        v = value_caches[layer_idx]
        v_compress_size, v_compress_time = v_cpu_compress(v, config)
        res["v_compress_size"].append(v_compress_size)
        res["v_compress_time"].append(v_compress_time)

    return res


def throughput_evaluation(
    config: PackKVCacheConfig,
    ctx_len: int,
    enable_save: bool,
    logger,
    collect_round: int = 1,
):
    logger.info(f"ctx_len: {ctx_len}")
    logger.info(config)
    logger.info(
        f"Created a temp config for cache collection and set its enable_quant to False."
    )
    PackKVCacheConfigStatic.config = PackKVCacheConfig(
        enable_quant=False,
        model_name=config.model_name,
        quant_method=config.quant_method,
        repack_method=config.repack_method,
        high_precision_zero_point=config.high_precision_zero_point,
        block_size=config.block_size,
        buffer_size=config.buffer_size,
        pack_size=config.pack_size,
        k_quant_scale_rel=config.k_quant_scale_rel,
        v_quant_scale_rel=config.v_quant_scale_rel,
    )

    PackKVCacheConfigStatic.extract_cache = ExtractCacheConfig(collect_round)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model_class = MODEL_CLASS_MAP.get(config.model_name)
    if model_class is None:
        raise ValueError(f"Model class not found for {config.model_name}")
    model = model_class.from_pretrained(
        PackKVCacheConfigStatic.config.model_name,
        torch_dtype="auto",
        device_map="auto",
        # attn_implementation="sdpa"
    )
    # batch_size = 1
    # lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)
    inputs = get_ctx_len_text_from_wikitext_103_v1(ctx_len, tokenizer).to(model.device)

    with torch.no_grad():
        _ = model(**inputs)
    PackKVCachePytorchQuant.round_ = 0

    if enable_save:
        save_extract_cache(
            config.model_name,
            ctx_len,
            PackKVCacheConfigStatic.extract_cache,
            "./dumped_cache",
        )

    rts = []

    logger.info(
        f"Cache Size {PackKVCacheConfigStatic.extract_cache.size() / 8 / 1024 / 1024 / 1024} GB"
    )

    for round_i in range(collect_round):
        key_caches = list(
            PackKVCacheConfigStatic.extract_cache.key_caches[round_i].values()
        )
        value_caches = list(
            PackKVCacheConfigStatic.extract_cache.value_caches[round_i].values()
        )
        rts.append(throughputs_evaluation_with_data(config, key_caches, value_caches))

    PackKVCacheConfigStatic.config = None
    PackKVCacheConfigStatic.extract_cache = None
    return rts


def eager_attention_forward(
    # query: torch.Tensor,
    key: List[torch.Tensor],
    value: List[torch.Tensor],
    # scaling: float,
):
    other_times = []
    b, h, c, hd = key[0].shape
    query = torch.randn(b, h, 1, hd, device=key[0].device, dtype=key[0].dtype)
    for key, value in zip(key, value):
        time = 0
        key = key.transpose(2, 3)

        attn_weights = torch.matmul(query, key)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        attn_weights = attn_weights * 11.31
        end.record()
        torch.cuda.synchronize()
        time += start.elapsed_time(end)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        end.record()
        torch.cuda.synchronize()
        time += start.elapsed_time(end)

        attn_weights = attn_weights.to(value.dtype)

        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous()
        other_times.append(time)
    return other_times


def throughput_evaluation_rebuttal(
    config: PackKVCacheConfig,
    ctx_len: int,
    enable_save: bool,
    logger,
    collect_round: int = 1,
):
    logger.info(f"ctx_len: {ctx_len}")
    logger.info(config)
    logger.info(
        f"Created a temp config for cache collection and set its enable_quant to False."
    )
    PackKVCacheConfigStatic.config = PackKVCacheConfig(
        enable_quant=False,
        model_name=config.model_name,
        quant_method=config.quant_method,
        repack_method=config.repack_method,
        high_precision_zero_point=config.high_precision_zero_point,
        block_size=config.block_size,
        buffer_size=config.buffer_size,
        pack_size=config.pack_size,
        k_quant_scale_rel=config.k_quant_scale_rel,
        v_quant_scale_rel=config.v_quant_scale_rel,
    )

    PackKVCacheConfigStatic.extract_cache = ExtractCacheConfig(collect_round)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model_class = MODEL_CLASS_MAP.get(config.model_name)
    if model_class is None:
        raise ValueError(f"Model class not found for {config.model_name}")
    model = model_class.from_pretrained(
        PackKVCacheConfigStatic.config.model_name,
        torch_dtype="auto",
        device_map="auto",
        # attn_implementation="sdpa"
    )
    # batch_size = 1
    # lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)
    real_ctx_len = 4 * 1024 if ctx_len >= 4 * 1024 else ctx_len
    assert ctx_len % real_ctx_len == 0
    ctx_len_factor = ctx_len // real_ctx_len
    inputs = get_ctx_len_text_from_wikitext_103_v1(real_ctx_len, tokenizer).to(
        model.device
    )

    with torch.no_grad():
        _ = model(**inputs)
    PackKVCachePytorchQuant.round_ = 0

    if enable_save:
        save_extract_cache(
            config.model_name,
            ctx_len,
            PackKVCacheConfigStatic.extract_cache,
            "./dumped_cache",
        )

    rts = []

    logger.info(
        f"Cache Size {PackKVCacheConfigStatic.extract_cache.size() / 8 / 1024 / 1024 / 1024 * ctx_len_factor} GB"
    )

    for round_i in range(collect_round):
        key_caches = list(
            PackKVCacheConfigStatic.extract_cache.key_caches[round_i].values()
        )
        value_caches = list(
            PackKVCacheConfigStatic.extract_cache.value_caches[round_i].values()
        )
        other_overhead = eager_attention_forward(key_caches, value_caches)
        rts.append(
            [
                throughputs_evaluation_with_data_rebuttal(
                    config, key_caches, value_caches, ctx_len_factor
                ),
                other_overhead,
            ]
        )

    PackKVCacheConfigStatic.config = None
    PackKVCacheConfigStatic.extract_cache = None
    return rts


def end_to_end_simulation_rebuttal(
    config: PackKVCacheConfig,
    eager_attention_speed_up: float,
    ctx_len: int,
    logger,
    batch_size: int = 1,
):
    logger.info(f"ctx_len: {ctx_len}")
    logger.info(config)
    model_name = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    logger.info(f"Creating {batch_size} prompts with target length {ctx_len} tokens...")
    base_prompt = "Hello, who are you? I am an AI assistant. "

    # First, tokenize base_prompt to get its actual token count
    base_tokens = tokenizer(base_prompt, return_tensors="pt").input_ids
    actual_tokens_per_repeat = base_tokens.shape[1]
    logger.info(f"Base prompt has {actual_tokens_per_repeat} tokens")

    # Calculate how many repetitions we need to reach ctx_len
    repeat_count = max(1, ctx_len // actual_tokens_per_repeat)
    logger.info(
        f"Repeating {repeat_count} times to reach ~{repeat_count * actual_tokens_per_repeat} tokens"
    )

    prompts = [base_prompt * repeat_count for _ in range(batch_size)]
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=ctx_len,
    ).to(model.device)

    actual_seq_len = inputs.input_ids.shape[1]
    logger.info(
        f"Created batch with shape: {inputs.input_ids.shape} (batch_size={batch_size}, seq_len={actual_seq_len})"
    )

    past_key_values = None
    current_input_ids = None
    cache_position = None

    def initialize_kv_cache():
        global past_key_values, current_input_ids, cache_position

        print(
            f"Creating StaticCache (batch_size={batch_size}, max_length={ctx_len + 100})..."
        )
        # Create static cache with room for prefill + decode tokens
        past_key_values = StaticCache(
            config=model.config,
            max_batch_size=batch_size,
            max_cache_len=ctx_len + 100,  # Extra room for decode tokens
            device=model.device,
            dtype=torch.float16,
        )

        print(f"Initializing KV cache with prefill (batch_size={batch_size})...")
        seq_len = inputs.input_ids.shape[1]
        cache_position = torch.arange(0, seq_len, device=model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
                logits_to_keep=1,  # solve prefill phase OOM
            )
            # past_key_values is updated in-place
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            current_input_ids = next_token_id
            # Update cache position for next decode step
            cache_position = torch.tensor([seq_len], device=model.device)

        logger.info(f"KV cache initialized. Current position: {cache_position.item()}")

    # Show GPU memory after prefill
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(
            f"GPU Memory after prefill: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved"
        )

    new_token = 10

    initialize_kv_cache()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", attn_implementation="eager"
    )

    def decode_func():
        global past_key_values, current_input_ids, cache_position
        generated_tokens = []

        with torch.no_grad():
            for i in range(new_token):
                outputs = model(
                    input_ids=current_input_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                )
                # past_key_values is updated in-place, no need to reassign
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                current_input_ids = next_token_id
                generated_tokens.append(next_token_id)
                # Increment cache position
                cache_position = cache_position + 1

        all_tokens = torch.cat(generated_tokens, dim=1)
        # Show first sample in batch
        decoded_text = tokenizer.decode(all_tokens[0], skip_special_tokens=True)
        print(f"Generated {new_token} tokens for batch_size={batch_size}")
        print(f"Sample from first item in batch: {decoded_text}")

    kernel_info, avg_func_time = profile_func(
        decode_func, warmup_runs=3, profile_runs=5
    )

    return kernel_info, avg_func_time
    # print("\nReturned kernel info:")
    # print(f"Number of kernels: {len(kernel_info['kernel_names'])}")
    # print(f"Total GPU time: {kernel_info['total_time']:.2f} us")
    # print(f"Time per token: {kernel_info['total_time'] / (batch_size * new_token):.2f} us")


def compressor_throughput_evaluation_rebuttal(
    config: PackKVCacheConfig,
    ctx_len: int,
    enable_save: bool,
    logger,
    collect_round: int = 1,
):
    logger.info(f"ctx_len: {ctx_len}")
    logger.info(config)
    logger.info(
        f"Created a temp config for cache collection and set its enable_quant to False."
    )
    PackKVCacheConfigStatic.config = PackKVCacheConfig(
        enable_quant=False,
        model_name=config.model_name,
        quant_method=config.quant_method,
        repack_method=config.repack_method,
        high_precision_zero_point=config.high_precision_zero_point,
        block_size=config.block_size,
        buffer_size=config.buffer_size,
        pack_size=config.pack_size,
        k_quant_scale_rel=config.k_quant_scale_rel,
        v_quant_scale_rel=config.v_quant_scale_rel,
    )

    PackKVCacheConfigStatic.extract_cache = ExtractCacheConfig(collect_round)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model_class = MODEL_CLASS_MAP.get(config.model_name)
    if model_class is None:
        raise ValueError(f"Model class not found for {config.model_name}")
    model = model_class.from_pretrained(
        PackKVCacheConfigStatic.config.model_name,
        torch_dtype="auto",
        device_map="auto",
        # attn_implementation="sdpa"
    )
    # batch_size = 1
    # lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)
    inputs = get_ctx_len_text_from_wikitext_103_v1(ctx_len, tokenizer).to(model.device)

    with torch.no_grad():
        _ = model(**inputs)
    PackKVCachePytorchQuant.round_ = 0

    if enable_save:
        save_extract_cache(
            config.model_name,
            ctx_len,
            PackKVCacheConfigStatic.extract_cache,
            "./dumped_cache",
        )

    rts = []

    logger.info(
        f"Cache Size {PackKVCacheConfigStatic.extract_cache.size() / 8 / 1024 / 1024 / 1024} GB"
    )
    size_mb = PackKVCacheConfigStatic.extract_cache.size() / 8 / 1024 / 1024

    for round_i in range(collect_round):
        key_caches = list(
            PackKVCacheConfigStatic.extract_cache.key_caches[round_i].values()
        )
        value_caches = list(
            PackKVCacheConfigStatic.extract_cache.value_caches[round_i].values()
        )
        rts.append(
            compressor_throughputs_evaluation_with_data(
                config, key_caches, value_caches
            )
        )

    PackKVCacheConfigStatic.config = None
    PackKVCacheConfigStatic.extract_cache = None
    return rts, size_mb


def cr_evaluation_detail_rebuttal(
    config: PackKVCacheConfig,
    ctx_len: int,
    enable_save: bool,
    logger,
    collect_round: int = 1,
    before_and_after_repacking=None,
):
    logger.info(f"ctx_len: {ctx_len}")
    logger.info(config)
    # logger.info(f"Created a temp config for cache collection and set its enable_quant to False.")
    PackKVCacheConfigStatic.config = PackKVCacheConfig(
        enable_quant=False,
        model_name=config.model_name,
        quant_method=config.quant_method,
        repack_method=config.repack_method,
        high_precision_zero_point=config.high_precision_zero_point,
        block_size=config.block_size,
        buffer_size=config.buffer_size,
        pack_size=config.pack_size,
        k_quant_scale_rel=config.k_quant_scale_rel,
        v_quant_scale_rel=config.v_quant_scale_rel,
    )

    PackKVCacheConfigStatic.extract_cache = ExtractCacheConfig(collect_round)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model_class = MODEL_CLASS_MAP.get(config.model_name)
    if model_class is None:
        raise ValueError(f"Model class not found for {config.model_name}")
    model = model_class.from_pretrained(
        PackKVCacheConfigStatic.config.model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    # batch_size = 1
    # lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)
    inputs = get_ctx_len_text_from_wikitext_103_v1(ctx_len, tokenizer).to(model.device)

    with torch.no_grad():
        _ = model(**inputs)
    PackKVCachePytorchQuant.round_ = 0

    if enable_save:
        save_extract_cache(
            config.model_name,
            ctx_len,
            PackKVCacheConfigStatic.extract_cache,
            "./dumped_cache",
        )

    rts = []

    logger.info(
        f"Cache Size {PackKVCacheConfigStatic.extract_cache.size() / 8 / 1024 / 1024 / 1024} GB"
    )

    for round_i in range(collect_round):
        key_caches = list(
            PackKVCacheConfigStatic.extract_cache.key_caches[round_i].values()
        )
        value_caches = list(
            PackKVCacheConfigStatic.extract_cache.value_caches[round_i].values()
        )
        rts.append(
            crs_evaluation_with_data_detail_rebuttal(
                config, key_caches, value_caches, before_and_after_repacking
            )
        )

    PackKVCacheConfigStatic.config = None
    PackKVCacheConfigStatic.extract_cache = None
    return rts


def repacking_throughput_rebuttal(
    config: PackKVCacheConfig,
    ctx_len: int,
    enable_save: bool,
    logger,
    collect_round: int = 1,
    before_and_after_repacking=None,
):
    logger.info(f"ctx_len: {ctx_len}")
    logger.info(config)
    # logger.info(f"Created a temp config for cache collection and set its enable_quant to False.")
    PackKVCacheConfigStatic.config = PackKVCacheConfig(
        enable_quant=False,
        model_name=config.model_name,
        quant_method=config.quant_method,
        repack_method=config.repack_method,
        high_precision_zero_point=config.high_precision_zero_point,
        block_size=config.block_size,
        buffer_size=config.buffer_size,
        pack_size=config.pack_size,
        k_quant_scale_rel=config.k_quant_scale_rel,
        v_quant_scale_rel=config.v_quant_scale_rel,
    )

    PackKVCacheConfigStatic.extract_cache = ExtractCacheConfig(collect_round)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model_class = MODEL_CLASS_MAP.get(config.model_name)
    if model_class is None:
        raise ValueError(f"Model class not found for {config.model_name}")
    model = model_class.from_pretrained(
        PackKVCacheConfigStatic.config.model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    # batch_size = 1
    # lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)
    inputs = get_ctx_len_text_from_wikitext_103_v1(ctx_len, tokenizer).to(model.device)

    with torch.no_grad():
        _ = model(**inputs)
    PackKVCachePytorchQuant.round_ = 0

    if enable_save:
        save_extract_cache(
            config.model_name,
            ctx_len,
            PackKVCacheConfigStatic.extract_cache,
            "./dumped_cache",
        )

    rts = []

    logger.info(
        f"Cache Size {PackKVCacheConfigStatic.extract_cache.size() / 8 / 1024 / 1024 / 1024} GB"
    )
    size_mb = PackKVCacheConfigStatic.extract_cache.size() / 8 / 1024 / 1024

    for round_i in range(collect_round):
        key_caches = list(
            PackKVCacheConfigStatic.extract_cache.key_caches[round_i].values()
        )
        value_caches = list(
            PackKVCacheConfigStatic.extract_cache.value_caches[round_i].values()
        )
        rts.append(
            repacking_throughput_with_kv_rebuttal(config, key_caches, value_caches)
        )

    PackKVCacheConfigStatic.config = None
    PackKVCacheConfigStatic.extract_cache = None
    return rts, size_mb
