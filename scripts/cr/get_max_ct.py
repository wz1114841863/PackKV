from transformers import AutoConfig, AutoTokenizer

model_path = "JackFram/llama-160m"

# 查看模型配置的最大长度
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print(
    f"Model config max length: {getattr(config, 'max_position_embeddings', 'Not Found')}"
)

# 查看分词器识别的最大长度
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f"Tokenizer max length: {tokenizer.model_max_length}")
