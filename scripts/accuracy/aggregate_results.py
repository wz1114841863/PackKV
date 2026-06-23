import os
import json
import glob
import pandas as pd
import argparse


def collect_results(base_dir):
    print(f"🔍 正在扫描目录: {base_dir}")
    # 递归寻找所有的 results.json 文件
    result_files = glob.glob(
        os.path.join(base_dir, "**", "results.json"), recursive=True
    )

    if not result_files:
        print("❌ 没有找到任何 results.json 文件,请检查路径.")
        return

    data = []
    for file_path in result_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                res = json.load(f)

            # 解析路径以提取配置参数
            # 路径结构: base_dir / model / task / quant_method / scale_X / ...
            parts = file_path.split(os.sep)

            scale_str = next(
                (p for p in parts if p.startswith("scale_")), "scale_unknown"
            )
            scale_val = scale_str.replace("scale_", "")

            # 提取算法名称 (在 scale_ 的上一层)
            try:
                scale_idx = parts.index(scale_str)
                quant_method = parts[scale_idx - 1]
            except ValueError:
                quant_method = "Unknown"

            results_dict = res.get("results", {})
            config_dict = res.get("config", {})
            model_name = config_dict.get("model_args", "Unknown").replace(
                "pretrained=", ""
            )

            for task_name, metrics in results_dict.items():
                # 不同任务的核心指标名称不一样:
                # MMLU 通常是 acc
                # GSM8K 通常是 exact_match
                # Hellaswag 通常是 acc_norm 或 acc
                acc = metrics.get(
                    "acc_norm,none",
                    metrics.get("exact_match,none", metrics.get("acc,none", None)),
                )

                # 兼容旧版本 lm-eval 的结构
                if acc is None:
                    acc = metrics.get(
                        "acc_norm", metrics.get("exact_match", metrics.get("acc", 0.0))
                    )

                if acc is not None:
                    acc = round(acc * 100, 2)

                data.append(
                    {
                        "Model": model_name,
                        "Task": task_name,
                        "Algorithm": quant_method,
                        "Scale": scale_val,
                        "Score (%)": acc,
                    }
                )
        except Exception as e:
            print(f"⚠️ 解析文件出错 {file_path}: {e}")

    # 生成 DataFrame 并排序
    df = pd.DataFrame(data)
    if not df.empty:
        # 按照 模型 -> 任务 -> 算法 -> Scale 排序,方便直观看到不同算法在不同Scale下的崩盘点
        df = df.sort_values(by=["Model", "Task", "Algorithm", "Scale"])

        print("\n" + "=" * 70)
        print("📊 PackKV 自动化消融实验结果汇总")
        print("=" * 70)
        print(df.to_markdown(index=False))

        out_csv = "summary_results.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n💾 详细结果已保存至: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="一键提取并汇总评测结果")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        required=True,
        help="例如: ./grid_search_logs/exp_0613_2300",
    )
    args = parser.parse_args()

    collect_results(args.dir)
