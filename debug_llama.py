import torch
from transformers import AutoTokenizer
from models.llama import LlamaForCausalLM
from models.cache.packkv_quant import PackKVCacheConfigStatic
from utils.config import PackKVCacheConfig
from utils.compute import QuantMethod, RepackMethod


def main():
    # model_id = "JackFram/llama-160m"
    model_id = "huggyllama/llama-7b"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"1. 初始化PackKV配置")
    packkv_config = PackKVCacheConfig(
        model_name=model_id,
        quant_method=QuantMethod.PackKV,
        repack_method=RepackMethod.MEDIAN,
        block_size=16,
        buffer_size=16,
        pack_size=4,
        k_quant_scale_rel=1.0,
        v_quant_scale_rel=1.0,
        enable_quant=True,
        high_precision_zero_point=False,
    )

    PackKVCacheConfigStatic.config = packkv_config
    PackKVCacheConfigStatic.extract_cache = None

    print(f"2. 加载模型和分词器")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(
        device
    )
    model.eval()

    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"3. 进行前向传播")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            use_cache=True,
        )

    print(f"4. 输出生成的文本")
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated Text: {generated_text}")


if __name__ == "__main__":
    main()
