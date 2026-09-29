import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.optimize import minimize
from hcipy import *
from flowfs import *
from tqdm import tqdm

parser = argparse.ArgumentParser(
                    prog='injetction_retrieval.py',
                    description='Calibrate the FLOWFS system from a sweep of zernike modes',
                    )
parser.add_argument('injection_file', type=str, help='Path to the sweep file (pickle)')
parser.add_argument('--from-config', type=str, default='./configs/flowfs_config_base.json', help='Path to initial FLOWFS config file (json)')
parser.add_argument('--num-workers', type=int, default=32, help='Number of workers for parallel processing')
parser.add_argument('--regularisation-strength', type=float, default=0.0, help='Regularisation strength for phase retrieval')
parser.add_argument('--num-modes', type=int, default=9, help='Number of upstream aberration modes to use')
parser.add_argument('--num-frames', type=int, default=5, help='Number of frames per poke to use')
parser.add_argument('--on-sky', action='store_true', help='Flag to indicate on-sky data. For on-sky data we only correct for downstream tip/tilt and other downstream aberrations are inferred from the starting config file.')
parser.add_argument('--plot', action='store_true', help='Flag to enable plotting of results.')

args = parser.parse_args()
injection_file = args.injection_file

flowfs = FLOWFS()
flowfs.load_config(args.from_config)

timestamp = injection_file[-23:-4]
data = pickle.load(open(injection_file, 'rb'))
num_modes = args.num_modes
flowfs.create_upstream_aberration(num_modes)
num_frames = args.num_frames
num_workers = args.num_workers
regularisation_strength = args.regularisation_strength
post_fix = "_on_sky" if args.on_sky else "_lab"

magaox_mask = make_magaox_bump_mask()
magellan_aperture = evaluate_supersampled(magaox_mask, flowfs.pupil_grid, 6)
wavefront = Wavefront(magellan_aperture, flowfs.wavelength_wfs)
wavefront.total_power = 1

# load data
im_refs = []
fs = []
acts = []

for i in range(len(data)):
    f, actuators, surface, images = data[i]

    im_refs.append(process_im(images, flowfs.binned_grid))
    acts.append(actuators)


im_refs = np.array(im_refs).reshape( 20, -1, 1024)
im_refs = im_refs.swapaxes(0,1)

acts = np.array(acts).reshape(20, -1, 9)
acts = acts.swapaxes(0,1)

ims = [Field(im, flowfs.binned_grid) for ims in im_refs for im in ims[:num_frames]]

flowfs.start_theta[:9] = flowfs.biases 
flowfs._set_aberration(flowfs.start_theta)
solver_ref_im, linear_solver, reconstruction_matrix = flowfs.get_reconstruction_matrix(wavefront, probe_amp=0.3, rcond=1e-2) # calculate around current NCPA

flowfs.start_theta[:9] = 0
flowfs._set_aberration(flowfs.start_theta)

results = phase_retrieval(flowfs, flowfs.start_theta.copy(), wavefront, ims[0], plot=args.plot, fix_downstream_tip_tilt=False, fit_downstream=False, regularisation_strength=regularisation_strength)

results = multi_phase_retrieval(flowfs, flowfs.start_theta.copy(), wavefront, ims, plot=False, fix_downstream_tip_tilt=False, fit_downstream=False, regularisation_strength=regularisation_strength, num_workers=num_workers)
fitted_params = np.array([r.x for r in results]).reshape(-1, num_frames, len(flowfs.start_theta))

fitted_params_linear = []
for i in (range(len(ims))):
    fitted_params_linear.append( linear_solver(ims[i]))

fitted_params_linear = np.array(fitted_params_linear).reshape(-1, num_frames, num_modes)

print(acts.shape, fitted_params.shape, fitted_params_linear.shape)

injection_reconstruction_plot(acts[:,0], fitted_params, fitted_params_linear, flowfs.zernike2dm_slopes, flowfs.biases, wavelength_wfs=flowfs.wavelength_wfs, save_name=f"./plots/injection_reconstruction_flowfs_{flowfs.mask_name}_{flowfs.stage_offset}mm_{flowfs.filter_name}_{num_modes}modes_{args.num_frames}frames_{'on_sky' if args.on_sky else 'lab'}_{regularisation_strength}_reg.png" if args.plot else None)
# plt.show()

os.makedirs(f'./results/{flowfs.mask_name}_{flowfs.stage_offset}mm_{flowfs.filter_name}_{num_modes}modes_{timestamp}{post_fix}', exist_ok=True)
np.save(f'./results/{flowfs.mask_name}_{flowfs.stage_offset}mm_{flowfs.filter_name}_{num_modes}modes_{timestamp}{post_fix}/injection_reconstruction_linear.npy', fitted_params_linear)
np.save(f'./results/{flowfs.mask_name}_{flowfs.stage_offset}mm_{flowfs.filter_name}_{num_modes}modes_{timestamp}{post_fix}/injection_reconstruction.npy', fitted_params)
np.save(f'./results/{flowfs.mask_name}_{flowfs.stage_offset}mm_{flowfs.filter_name}_{num_modes}modes_{timestamp}{post_fix}/injection_acts.npy', acts)
