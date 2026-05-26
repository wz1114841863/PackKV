# PackKV
PackKV官方代码学习

## 环境配置
```
conda config --set ssl_verify false
conda create -n debug_env python=3.12 -y
conda activate debug_env
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit -y
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install -r ./requirements.txt
MAX_JOBS=8 uv pip install flash-attn --no-build-isolation
cd packkv_cuda_ext
uv pip install -e . --no-build-isolation
cd ..
```

## 代码结构说明
- **`packkv_cuda_ext/`**: C++ and CUDA source code for the packkv custom extension.
- **`models/`**: implementations of LLMs used in PackKV Experiments.
- **`scripts/`**: Scripts for automation, benchmarking, and generating experimental results.
- **`evaluation/`**: Code related to model evaluation.
- **`utils/`**: Helper functions and utilities.

```

```
