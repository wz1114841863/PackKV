cd scripts
export PYTHONPATH=~/PackKV

cd figure
echo "==================== Figure 3, 4 ===================="
CUDA_VISIBLE_DEVICES=0 python kv_data_characteristic.py # < 1 min
echo "==================== Figure 3, 4 ===================="

echo "==================== Figure 6 ===================="
CUDA_VISIBLE_DEVICES=0 python repacking_visualization.py # < 1 min
echo "==================== Figure 6 ===================="

cd ..
mkdir data
mkdir data/pack_size_cr
echo "==================== Figure 13 ===================="
CUDA_VISIBLE_DEVICES=0 python pack_size_cr_gen_setting.py # < 1 min
CUDA_VISIBLE_DEVICES=0 python pack_size_cr_run.py # < 20 mins
CUDA_VISIBLE_DEVICES=0 python figure/pack_size_cr.py # < 1 min
echo "==================== Figure 13 ===================="

mkdir data/accuracy
mkdir turning_point
mkdir figure/accuracy_turning_point
echo "==================== Figure 14 ===================="
python accuracy_gen_setting.py # < 1 min
python result_filter.py --setting data/accuracy/accuracy_setting_map.pkl --result data/accuracy/accuracy_result_map.pkl # < 1 min
python accuracy_run.py # < 150 hours with two RTX Pro 6000, < 300 hours with one RTX Pro 6000
python turning_point_from_accuracy.py # < 1 min
python figure/accuracy_w_fitting.py # < 1 min
echo "==================== Figure 14 ===================="

mkdir data/throughput
echo "==================== Figure 15, 16 ===================="
CUDA_VISIBLE_DEVICES=0 python throughput_gen_setting_rtx_pro.py # < 1 min
CUDA_VISIBLE_DEVICES=0 python throughput_run_rtx_pro.py # < 10 mins
CUDA_VISIBLE_DEVICES=0 python figure/throughput_rtx_pro.py # < 1 min
echo "==================== Figure 15, 16 ===================="

# Figure 17:
# echo "==================== Figure 17 ===================="
# python throughput_multi_gpu_scaling_gen_setting_a100.py
# python throughput_multi_gpu_scaling_run_a100.py
# python figure/throughput_multi_gpu.py
# echo "==================== Figure 17 ===================="

echo "==================== Table 1 ===================="
python table/lossless_cr_table.py # < 1 min
echo "==================== Table 1 ===================="

echo "==================== Table 3, 4 ===================="
python table/turning_point_quant_scale_table.py # < 1 min
echo "==================== Table 3, 4 ===================="

echo "==================== Table 2, 5 ===================="
python turning_point_cr_gen_setting.py # < 1 min
python result_filter.py --setting data/turning_point/turning_point_cr_setting_map.pkl --result data/turning_point/turning_point_cr_result_map.pkl # < 1 min
python turning_point_cr_run.py # < 30 mins
python turning_point_cr_result_append.py # < 1 min
python table/turning_point_cr_table.py # < 1 min
echo "==================== Table 2, 5 ===================="