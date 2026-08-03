import ast
import math

from utils.compute import (
    BucketScoreMethod,
    QuantMode,
    QuantMethod,
    RepackMethod,
    ScaleMethod,
)


class ExtractCacheConfig:
    """数据分布探针, 获取模型内部的KV Cache数据, 用于分析和调试"""

    def __init__(self, collect_round: int):
        self.collect_round = collect_round
        self.key_caches = {}
        self.value_caches = {}

    def size(self):
        total_size = 0
        # {轮次ID: {层级ID或头部ID: 张量数据}}
        for round_ks in self.key_caches.values():
            for k in round_ks.values():
                total_size += k.numel() * 16

        for round_vs in self.value_caches.values():
            for v in round_vs.values():
                total_size += v.numel() * 16

        return total_size


class PackKVCacheConfig:
    def __init__(
        self,
        model_name: str,
        quant_method: QuantMethod,  # 量化方法
        repack_method: RepackMethod,  # 重排策略
        high_precision_zero_point: bool,  # 决定零点元数据是否暴露高精度
        block_size: int,  # 多少个Token被切分为一个基础Block
        buffer_size: int,  # Recent Window, 保留多少个最近的Token在Buffer中不压缩
        pack_size: int,  # 在重排后，几个 Token 被打包在一起计算共用位宽
        k_quant_scale_rel: float,  # 相对缩放比例, 量化的时候决定K Cache的量化步长, 拉大这个值可以制造更大的误差
        v_quant_scale_rel: float,
        # enable_k_minus_avg: bool,
        enable_quant: bool = True,
        scale_method: ScaleMethod = ScaleMethod.CONTINUOUS,
        bucket_count: int = 4,
        bucket_score_method: BucketScoreMethod = BucketScoreMethod.COMBINED_SUM,
        k_error_budget: float = 0.1,
        v_error_budget: float = 0.1,
    ):
        if not math.isfinite(k_error_budget) or k_error_budget < 0:
            raise ValueError("k_error_budget must be finite and non-negative")
        if not math.isfinite(v_error_budget) or v_error_budget < 0:
            raise ValueError("v_error_budget must be finite and non-negative")
        self.enable_quant: bool = enable_quant
        self.model_name: str = model_name
        self.quant_method: QuantMethod = quant_method
        self.repack_method: RepackMethod = repack_method
        self.high_precision_zero_point: bool = high_precision_zero_point
        self.block_size: int = block_size
        self.buffer_size: int = buffer_size
        self.pack_size: int = pack_size
        # self.enable_k_minus_avg: bool = enable_k_minus_avg
        self.k_quant_scale_rel: float = k_quant_scale_rel
        self.v_quant_scale_rel: float = v_quant_scale_rel
        self.scale_method: ScaleMethod = scale_method
        self.bucket_count: int = bucket_count
        self.bucket_score_method: BucketScoreMethod = bucket_score_method
        self.k_error_budget: float = k_error_budget
        self.v_error_budget: float = v_error_budget

    # to string print
    def __str__(self):
        # parse class as json
        json_ = {
            "enable_quant": self.enable_quant,
            "model_name": self.model_name,
        }
        if self.enable_quant:
            # json_["enable_k_minus_avg"] = self.enable_k_minus_avg
            json_["quant_method"] = self.quant_method.name
            json_["repack_method"] = self.repack_method.value
            json_["high_precision_zero_point"] = self.high_precision_zero_point
            json_["block_size"] = self.block_size
            json_["buffer_size"] = self.buffer_size
            json_["pack_size"] = self.pack_size
            json_["k_quant_scale_rel"] = self.k_quant_scale_rel
            json_["v_quant_scale_rel"] = self.v_quant_scale_rel
            json_["scale_method"] = getattr(
                self, "scale_method", ScaleMethod.CONTINUOUS
            ).value
            json_["bucket_count"] = getattr(self, "bucket_count", 4)
            json_["bucket_score_method"] = getattr(
                self,
                "bucket_score_method",
                BucketScoreMethod.COMBINED_SUM,
            ).value
            json_["k_error_budget"] = getattr(self, "k_error_budget", 0.1)
            json_["v_error_budget"] = getattr(self, "v_error_budget", 0.1)

        return str(json_)

    @staticmethod
    def from_str(json_str: str):
        json_ = ast.literal_eval(json_str)
        if json_["enable_quant"]:
            quant_method_raw = json_["quant_method"]
            if isinstance(quant_method_raw, str):
                quant_method = QuantMethod[quant_method_raw]
            else:
                quant_method = QuantMethod(quant_method_raw)
            repack_method = RepackMethod(json_["repack_method"])
            high_precision_zero_point = json_["high_precision_zero_point"]
            block_size = json_["block_size"]
            buffer_size = json_["buffer_size"]
            pack_size = json_["pack_size"]
            k_quant_scale_rel = json_["k_quant_scale_rel"]
            v_quant_scale_rel = json_["v_quant_scale_rel"]
            scale_method = ScaleMethod(
                json_.get("scale_method", ScaleMethod.CONTINUOUS.value)
            )
            bucket_count = json_.get("bucket_count", 4)
            bucket_score_method = BucketScoreMethod(
                json_.get(
                    "bucket_score_method",
                    BucketScoreMethod.COMBINED_SUM.value,
                )
            )
            k_error_budget = json_.get("k_error_budget", 0.1)
            v_error_budget = json_.get("v_error_budget", 0.1)
        else:
            quant_method = None
            repack_method = None
            high_precision_zero_point = False
            block_size = None
            buffer_size = None
            pack_size = None
            k_quant_scale_rel = None
            v_quant_scale_rel = None
            scale_method = ScaleMethod.CONTINUOUS
            bucket_count = 4
            bucket_score_method = BucketScoreMethod.COMBINED_SUM
            k_error_budget = 0.1
            v_error_budget = 0.1

        return PackKVCacheConfig(
            model_name=json_["model_name"],
            enable_quant=json_["enable_quant"],
            # enable_k_minus_avg=json_["enable_k_minus_avg"],
            quant_method=quant_method,
            repack_method=repack_method,
            high_precision_zero_point=high_precision_zero_point,
            block_size=block_size,
            buffer_size=buffer_size,
            pack_size=pack_size,
            k_quant_scale_rel=k_quant_scale_rel,
            v_quant_scale_rel=v_quant_scale_rel,
            scale_method=scale_method,
            bucket_count=bucket_count,
            bucket_score_method=bucket_score_method,
            k_error_budget=k_error_budget,
            v_error_budget=v_error_budget,
        )

    def __eq__(self, other):
        if not isinstance(other, PackKVCacheConfig):
            return False
        if self.enable_quant != other.enable_quant:
            return False
        if self.model_name != other.model_name:
            return False
        # if self.enable_k_minus_avg != other.enable_k_minus_avg:
        #     return False
        if self.quant_method != other.quant_method:
            return False
        if self.repack_method != other.repack_method:
            return False
        if self.high_precision_zero_point != other.high_precision_zero_point:
            return False
        if self.block_size != other.block_size:
            return False
        if self.buffer_size != other.buffer_size:
            return False
        if self.pack_size != other.pack_size:
            return False
        if self.k_quant_scale_rel != other.k_quant_scale_rel:
            return False
        if self.v_quant_scale_rel != other.v_quant_scale_rel:
            return False
        if getattr(
            self, "scale_method", ScaleMethod.CONTINUOUS
        ) != getattr(other, "scale_method", ScaleMethod.CONTINUOUS):
            return False
        if getattr(self, "bucket_count", 4) != getattr(other, "bucket_count", 4):
            return False
        if getattr(
            self,
            "bucket_score_method",
            BucketScoreMethod.COMBINED_SUM,
        ) != getattr(
            other,
            "bucket_score_method",
            BucketScoreMethod.COMBINED_SUM,
        ):
            return False
        if getattr(self, "k_error_budget", 0.1) != getattr(
            other, "k_error_budget", 0.1
        ):
            return False
        if getattr(self, "v_error_budget", 0.1) != getattr(
            other, "v_error_budget", 0.1
        ):
            return False
        return True

    def __hash__(self):
        return hash(
            (
                self.enable_quant,
                self.model_name,
                # self.enable_k_minus_avg,
                self.quant_method,
                self.repack_method,
                self.high_precision_zero_point,
                self.block_size,
                self.buffer_size,
                self.pack_size,
                self.k_quant_scale_rel,
                self.v_quant_scale_rel,
                getattr(self, "scale_method", ScaleMethod.CONTINUOUS),
                getattr(self, "bucket_count", 4),
                getattr(
                    self,
                    "bucket_score_method",
                    BucketScoreMethod.COMBINED_SUM,
                ),
                getattr(self, "k_error_budget", 0.1),
                getattr(self, "v_error_budget", 0.1),
            )
        )

    # for print
    def __repr__(self):
        return self.__str__()
