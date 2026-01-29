import argparse
from utils.serialization import load, save

parser = argparse.ArgumentParser(description="Filter result_map by setting_map and save the filtered results.")
parser.add_argument('--setting', type=str, required=True, help='Path to the setting_map pickle file')
parser.add_argument('--result', type=str, required=True, help='Path to the result_map pickle file')
parser.add_argument('--output', type=str, default=None, help='Path to save the filtered result_map (default: overwrite result_map)')
args = parser.parse_args()

setting_path = args.setting
setting_map = load(setting_path)

result_path = args.result
result_map = load(result_path)

new_result_map = {}
for benchmark, config in setting_map.items():
    if benchmark in result_map and not isinstance(result_map[benchmark], str):
        new_result_map[benchmark] = result_map[benchmark]

new_result_path = args.output if args.output else result_path
save(new_result_map, new_result_path)
print(f"Found {len(new_result_map)} results reusable")