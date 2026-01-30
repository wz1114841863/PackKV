cd scripts
export PYTHONPATH=~/PackKV

mkdir data
mkdir data/throughput
echo "==================== Figure 15, 16 ===================="
python throughput_gen_setting_a100.py
python throughput_run_a100.py
python figure/throughput_a100.py
echo "==================== Figure 15, 16 ===================="

echo "==================== Figure 17 ===================="
python throughput_multi_gpu_scaling_gen_setting_a100.py
python throughput_multi_gpu_scaling_run_a100.py --gpu_count 1
python throughput_multi_gpu_scaling_run_a100.py --gpu_count 2
python throughput_multi_gpu_scaling_run_a100.py --gpu_count 3
python throughput_multi_gpu_scaling_run_a100.py --gpu_count 4
python figure/throughput_multi_gpu.py
echo "==================== Figure 17 ===================="