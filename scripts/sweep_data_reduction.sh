#!/usr/bin/env bash

NUM_FRAMES=2
NUM_WORKERS=12

# Lab sweeps 2025-11-30 (start from base configs to generate *_lab.json)
python examples/calibration_from_sweep.py data/flowfs_sweep_lyotlg_14.0mm_z_9modes_2025-11-30_12_34_13.pkl --from-config ./configs/flowfs_config_lyotlg.json --num-frames "$NUM_FRAMES" --num-workers "$NUM_WORKERS" 
# python examples/calibration_from_sweep.py data/flowfs_sweep_knifemaskZ_14.0mm_z_9modes_2025-11-30_12_24_25.pkl --from-config ./configs/flowfs_config_base_knifemaskZ.json --num-frames "$NUM_FRAMES" --num-workers "$NUM_WORKERS" 
# python examples/calibration_from_sweep.py data/flowfs_sweep_knifemask_14.0mm_z_9modes_2025-11-30_12_15_36.pkl --from-config ./configs/flowfs_config_base_knifemask.json --num-frames "$NUM_FRAMES" --num-workers "$NUM_WORKERS" 
# python examples/calibration_from_sweep.py data/flowfs_sweep_lyotsm_14.0mm_z_9modes_2025-11-30_12_05_58.pkl --from-config ./configs/flowfs_config_lyot_sm.json --num-frames "$NUM_FRAMES" --num-workers "$NUM_WORKERS" 
# python examples/calibration_from_sweep.py data/flowfs_sweep_lyotsm_14.0mm_r_9modes_2025-11-30_12_10_46.pkl --from-config ./configs/flowfs_config_lyot_sm_r.json --num-frames "$NUM_FRAMES" --num-workers "$NUM_WORKERS" 


# On-sky sweeps (use corresponding lab configs for reconstruction)
# python examples/calibration_from_sweep.py data/flowfs_sweep_lyotlg_14.0mm_z_9modes_2025-12-02_00_16_06.pkl --num-frames "$NUM_FRAMES" --num-workers "$NUM_WORKERS" --on-sky --from-config configs/flowfs_lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab.json
# python examples/calibration_from_sweep.py data/flowfs_sweep_knifemaskZ_14.0mm_z_9modes_2025-12-02_23_02_09.pkl  --num-frames "$NUM_FRAMES" --num-workers "$NUM_WORKERS" --on-sky --from-config configs/flowfs_knifemaskZ_14mm_z_9modes_2025-11-30_12_24_25_lab.json
# python examples/calibration_from_sweep.py "data/flowfs_sweep_lyotsm_14.0mm_z_9modes_2025-04-15_02_21_15.pkl"  --num-frames "$NUM_FRAMES" --num-workers "$NUM_WORKERS" --on-sky --from-config configs/flowfs_lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab.json --plot
