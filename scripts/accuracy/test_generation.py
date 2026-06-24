import torch
import time
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.config import PackKVCacheConfig, ExtractCacheConfig
from models.cache.packkv_quant import PackKVCacheConfigStatic, PackKVCachePytorchQuant
from utils.compute import (
    QuantMode,
    QuantMethod,
    RepackMethod,
    quant_ints,
    repack_and_encode,
    repack_and_encode_detail_rebuttal,
    repack_throughput_detail_rebuttal,
    quant_ints_throughput,
)
from models.llama import LlamaForCausalLM
from models.phi import Phi3ForCausalLM
from models.qwen3 import Qwen3ForCausalLM
from models.mistral import MistralForCausalLM

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
    "mistralai/Ministral-3-8B-Instruct-2512": MistralForCausalLM,
    "microsoft/phi-4": Phi3ForCausalLM,
}


def debug_simple_generation(config):
    print("=" * 50)
    print(f"[Debug] 正在初始化极简生成测试...")
    print(f"[Debug] 模型: {config.model_name}")
    print(
        f"[Debug] 量化配置: K_scale={config.k_quant_scale_rel}, V_scale={config.v_quant_scale_rel}"
    )
    print("=" * 50)

    PackKVCacheConfigStatic.config = config
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_class = MODEL_CLASS_MAP.get(config.model_name, AutoModelForCausalLM)
    print(f"[Debug] 正在加载模型类: {model_class.__name__} ...")
    model = model_class.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # 准备极简测试 Prompt
    # 用一道非常简单的常识题或数学题
    prompt = "Question: What is 15 + 27?\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"\n[Debug] 输入 Prompt: '{prompt}'")
    print(f"[Debug] 开始生成 ...")

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,  # 限制极短的生成长度,坏了也不用等太久
            do_sample=False,  # 关闭采样,使用贪心策略保证结果稳定
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    end_time = time.time()

    # 解码并打印结果
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n" + "=" * 50)
    print("[生成结果]:")
    print(generated_text)
    print("=" * 50)
    print(f"[Debug] 生成耗时: {end_time - start_time:.2f} 秒")

    PackKVCacheConfigStatic.config = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PackKV 本地极简生成测试脚本")

    # 定义可输入的命令行参数
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="Qwen/Qwen3-3B",
        help="需要测试的模型名称",
    )
    parser.add_argument(
        "-k",
        "--k_scale",
        type=float,
        default=0.01,
        help="K Cache量化Scale",
    )
    parser.add_argument(
        "-v",
        "--v_scale",
        type=float,
        default=0.01,
        help="V Cache量化Scale",
    )

    args = parser.parse_args()

    BLOCK_SIZE = 64
    BUFFER_SIZE = 128 + 64

    # 构造配置,接收来自命令行的参数
    config = PackKVCacheConfig(
        enable_quant=True,
        model_name=args.model_name,
        quant_method=QuantMethod.KIVI,
        repack_method=RepackMethod.NONE,
        high_precision_zero_point=False,
        block_size=BLOCK_SIZE,
        buffer_size=BUFFER_SIZE,
        pack_size=16,
        k_quant_scale_rel=args.k_scale,
        v_quant_scale_rel=args.v_scale,
    )

    debug_simple_generation(config)
