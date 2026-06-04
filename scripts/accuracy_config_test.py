#! /usr/bin/env python
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluation import accuracy_evaluation
from utils.compute import QuantMethod, RepackMethod
from utils.config import PackKVCacheConfig
from utils.util import get_logger, block_other_logger, register_notify

logger = get_logger(__file__)
block_other_logger(logger)
# register_notify()

model_name = "Qwen/Qwen3-4B"
BLOCK_SIZE = 64
BUFFER_SIZE = 128 + 64

config = PackKVCacheConfig(
    enable_quant=True,
    model_name=model_name,
    quant_method=QuantMethod.PackKV,  # 在这里切换 KIVI 或 PackKV 测试
    repack_method=RepackMethod.NONE,
    high_precision_zero_point=False,
    block_size=BLOCK_SIZE,
    buffer_size=BUFFER_SIZE,
    pack_size=16,
    k_quant_scale_rel=0.07,  # 拉大 K Cache 的量化步长,制造误差
    v_quant_scale_rel=0.01,  # 锁死 V Cache 排除干扰
)

benchmark = "winogrande"
print(accuracy_evaluation(config=config, benchmark=benchmark, logger=logger))

benchmark = "winogrande"

print(accuracy_evaluation(config=config, benchmark=benchmark, logger=logger))

# config.k_quant_scale_rel=0.07
#
# print(accuracy_evaluation(
#     config=config,
#     benchmark=benchmark,
#     logger=logger
# ))

# 0.6093333333333334
