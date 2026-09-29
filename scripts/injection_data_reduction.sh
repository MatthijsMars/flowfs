#!/usr/bin/env bash

NUM_FRAMES=20
NUM_WORKERS=10

# Lab injections (use sweep-derived lab configs)
python examples/injection_retrieval.py data/flowfs_random_lyotlg_14.0mm_z_9modes_2025-11-30_12_34_42.pkl --from-config configs/flowfs_lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab.json  --num-workers "$NUM_WORKERS" --num-frames "$NUM_FRAMES"
python examples/injection_retrieval.py data/flowfs_random_knifemaskZ_14.0mm_z_9modes_2025-11-30_12_24_53.pkl --from-config configs/flowfs_knifemaskZ_14mm_z_9modes_2025-11-30_12_24_25_lab.json  --num-workers "$NUM_WORKERS" --num-frames "$NUM_FRAMES"
python examples/injection_retrieval.py data/flowfs_random_knifemask_14.0mm_z_9modes_2025-11-30_12_16_03.pkl --from-config configs/flowfs_knifemask_14mm_z_9modes_2025-11-30_12_15_36_lab.json  --num-workers "$NUM_WORKERS" --num-frames "$NUM_FRAMES"
python examples/injection_retrieval.py data/flowfs_random_lyotsm_14.0mm_z_9modes_2025-11-30_12_07_21.pkl --from-config configs/flowfs_lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab.json  --num-workers "$NUM_WORKERS" --num-frames "$NUM_FRAMES"
python examples/injection_retrieval.py data/flowfs_random_lyotsm_14.0mm_r_9modes_2025-11-30_12_11_14.pkl --from-config configs/flowfs_lyot_sm_14mm_r_9modes_2025-11-30_12_10_46_lab.json  --num-workers "$NUM_WORKERS" --num-frames "$NUM_FRAMES"

# On-sky injections (use sweep-derived on-sky configs)
python examples/injection_retrieval.py data/flowfs_random_lyotlg_14.0mm_z_9modes_2025-12-02_00_16_33.pkl --from-config configs/flowfs_lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky.json --on-sky  --num-workers "$NUM_WORKERS" --num-frames "$NUM_FRAMES"
# python examples/injection_retrieval.py data/flowfs_random_knifemaskZ_14.0mm_z_9modes_2025-12-02_23_02_17.pkl --from-config configs/flowfs_knifemaskZ_14mm_z_9modes_2025-12-02_23_02_09_on_sky.json --on-sky  --num-workers "$NUM_WORKERS" --num-frames "$NUM_FRAMES"
# python examples/injection_retrieval.py data/flowfs_random_lyot_sm_14.0mm_z_9modes_2025-04-15_02_21_15.pkl --from-config configs/flowfs_lyot_sm_14mm_z_9modes_2025-04-15_02_21_15_on_sky.json --on-sky  --num-workers "$NUM_WORKERS" --num-frames "$NUM_FRAMES"