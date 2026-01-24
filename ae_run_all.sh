cd scripts
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.. 

# Figure 3, 4: 
python figure/kv_data_characteristic.py

# Figure 6: 
python figure/repacking_visualization.py

# Figure 13: 
- scripts/pack_size_cr_gen_setting.py
- scripts/pack_size_cr_run.py
- scripts/figure/pack_size_cr.py

Figure 14:
- scripts/accuracy_gen_setting.py
- scripts/accuracy_run.py
- scripts/turning_point_from_accuracy.py
- scripts/figure/accuracy_w_fitting.py

Figure 15, 16:
- scripts/throughput_gen_setting_a100.py
- scripts/throughput_run_a100.py
- scripts/throughput_gen_setting_rtx_pro.py
- scripts/throughput_run_rtx_pro.py
- scripts/figure/throughput.py

Figure 17:
- scripts/throughput_multi_gpu_scaling_gen_setting_a100.py
- scripts/throughput_multi_gpu_scaling_run_a100.py
- scripts/figure/throughput_multi_gpu.py

Table 1:
- scripts/pack_size_cr_gen_setting.py
- scripts/pack_size_cr_run.py
- scripts/table/lossless_cr_table.py

Table 2:
- scripts/table/turning_point_cr_table.py

Table 3, 4:
- scripts/table/turning_point_quant_scale_table.py

Table 5:
- scripts/table/turning_point_cr_table.py