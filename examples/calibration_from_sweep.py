import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"


import pickle
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from hcipy import *
from flowfs import *
from tqdm import tqdm

parser = argparse.ArgumentParser(
                    prog='calibration_from_sweep.py',
                    description='Calibrate the FLOWFS system from a sweep of zernike modes',
                    )
parser.add_argument('sweep_file', type=str, help='Path to the sweep file (pickle)')
parser.add_argument('--from-config', type=str, default='./configs/flowfs_config_base.json', help='Path to initial FLOWFS config file (json)')
parser.add_argument('--num-workers', type=int, default=32, help='Number of workers for parallel processing')
parser.add_argument('--regularisation-strength', type=float, default=0.0, help='Regularisation strength for phase retrieval')
parser.add_argument('--num-modes', type=int, default=9, help='Number of upstream aberration modes to use')
parser.add_argument('--num-frames', type=int, default=5, help='Number of frames per poke to use')
parser.add_argument('--on-sky', action='store_true', help='Flag to indicate on-sky data. For on-sky data we only correct for downstream tip/tilt and other downstream aberrations are inferred from the starting config file.')
parser.add_argument('--plot', action='store_true', help='Flag to enable plotting of results.')

args = parser.parse_args()


flowfs = FLOWFS()
flowfs.load_config(args.from_config)

if flowfs.start_theta[-6] == 0:
    flowfs.start_theta[-6] = 18 # set an initial defocus

sweep_file = args.sweep_file
data = pickle.load(open(sweep_file, 'rb'))
timestamp = sweep_file[-23:-4]
num_modes = args.num_modes
flowfs.create_upstream_aberration(num_modes)
num_acts = 11 # amount of different actuator pokes
num_frames = args.num_frames
num_workers = args.num_workers
regularisation_strength =  args.regularisation_strength
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

    im_refs.append(images)
    acts.append(actuators)

a = np.array(im_refs).reshape(-1, 9, 11, 1024)
a = a.swapaxes(0,1).swapaxes(1,2)

b = np.array(acts).reshape(-1, 9, 11, 9)
b = b.swapaxes(0,1).swapaxes(1,2)

ims = a[:num_modes,:,:num_frames].reshape(-1, 1024)
ims = [Field(im, flowfs.binned_grid) for im in ims]

if flowfs.mask_name.startswith('knifemask'):
    num_steps = 30
    frame = 2
    if flowfs.mask_name == 'knifemaskZ':
        offsets = np.linspace(0, 3, num_steps)
    elif flowfs.mask_name == 'knifemask':
        offsets = np.linspace(-3, 0, num_steps)
    loss = []
    for offset in tqdm(offsets):
        flowfs.knife_edge_offset = [0, offset]
        flowfs._set_mask(flowfs.mask_name)
        result = phase_retrieval(flowfs, flowfs.start_theta.copy(), wavefront, im_refs[5*num_frames], plot=False, fix_downstream_tip_tilt=True, fit_downstream=False, regularisation_strength=args.regularisation_strength)

        loss.append(result.fun)

    if args.plot:
        plt.plot(offsets, loss)
        plt.show(block=False)

    best_offset = offsets[np.argmin(loss)]
    print("Best offset:", best_offset)
    flowfs.knife_edge_offset = [0, best_offset]
    flowfs._set_mask(flowfs.mask_name)
    result = phase_retrieval(flowfs, flowfs.start_theta.copy(), wavefront, im_refs[5], plot=args.plot, fix_downstream_tip_tilt=True, fit_downstream=False, regularisation_strength=args.regularisation_strength)
    print(result.x)


# First fit the 0-points to determine downstream NCPA
fitted_params = []
fit_downstream = False if args.on_sky else True

results = multi_phase_retrieval(flowfs, flowfs.start_theta.copy(), wavefront, ims, plot=False, fix_downstream_tip_tilt=True, fit_downstream=fit_downstream, regularisation_strength=regularisation_strength, num_workers=num_workers)
fitted_params = np.array([r.x for r in results]).reshape(-1, num_frames, len(flowfs.start_theta))

flowfs.start_theta[-8:] = np.mean(fitted_params, axis=(0,1))[-8:]
flowfs.fitted_parameters[:] = True
flowfs._set_aberration(flowfs.start_theta)

if args.num_modes != 9:
    num_modes = 9
    # refit the upstream bias with 9 modes as that is what we use for the reconstructor
    flowfs.create_upstream_aberration(num_modes) # always do 9 modes for the sweep and calculation of the reconstruction matrix
    results = multi_phase_retrieval(flowfs, flowfs.start_theta.copy(), wavefront, ims, plot=False, fix_downstream_tip_tilt=False, fit_downstream=fit_downstream, regularisation_strength=regularisation_strength, num_workers=num_workers)
    fitted_params = np.array([r.x for r in results]).reshape(-1, num_frames, len(flowfs.start_theta))

p = np.copy(fitted_params).reshape(9, 11, -1, len(flowfs.start_theta))
rms = lambda x: np.sqrt(np.mean(np.square(x)))
bias = np.mean(p[:,5], axis=(0,1))[:]
print("mean non-injected rms", 908/(2*np.pi)*np.mean([rms(x[:9]-bias[:9]) for x in p[:,5].reshape(-1, len(flowfs.start_theta))]))

flowfs.start_theta[:] = np.mean(fitted_params, axis=(0,1))[:] # set start point to the last fit

solver_ref_im, linear_solver, reconstruction_matrix = flowfs.get_reconstruction_matrix(wavefront, probe_amp=0.3, rcond=1e-2) # calculate around current NCPA

flowfs.start_theta[:-8] = 0 # set upstream to 0 for non-linear reconstructor
flowfs.fitted_parameters[:] = True
flowfs._set_aberration(flowfs.start_theta)

flowfs.save_config(f'configs/flowfs_{flowfs.mask_name}_{flowfs.stage_offset}mm_{flowfs.filter_name}_{num_modes}modes_{timestamp}{post_fix}.json')



res = phase_retrieval(flowfs, flowfs.start_theta.copy(), wavefront, ims[5], plot=args.plot, fix_downstream_tip_tilt=False, fit_downstream=False)

ims = a[:num_modes,:,:num_frames].reshape(-1, 1024)
ims = [Field(im, flowfs.binned_grid) for im in ims]

sweep_params = []

results = multi_phase_retrieval(flowfs, flowfs.start_theta.copy(), wavefront, ims, plot=False, fix_downstream_tip_tilt=False, fit_downstream=False, regularisation_strength=regularisation_strength, num_workers=num_workers)
sweep_params = np.array([r.x for r in results]).reshape(-1, num_frames, len(flowfs.start_theta))

# calculate biases to use for the linear reconstructor
slopes = None if not args.on_sky else flowfs.zernike2dm_slopes
biases, slopes = plot_sweep_combined(acts, np.array(sweep_params), np.array(sweep_params), slopes=slopes)
plt.clf()

flowfs.start_theta[:-8] = biases # set upstream to the bias
flowfs.fitted_parameters[:] = True
flowfs._set_aberration(flowfs.start_theta)
solver_ref_im, linear_solver, reconstruction_matrix = flowfs.get_reconstruction_matrix(wavefront, probe_amp=0.3, rcond=1e-2) # calculate around current NCPA

sweep_params_linear = []
for i in (range(len(ims[:num_modes * num_acts * num_frames]))):
    sweep_params_linear.append( linear_solver(ims[i]))

sweep_params_linear = np.array(sweep_params_linear).reshape(-1, num_frames, num_modes)

slopes = None if not args.on_sky else flowfs.zernike2dm_slopes
biases, slopes = plot_sweep_combined(acts, np.array(sweep_params), np.array(sweep_params_linear), save_name=f"./plots/sweep_flowfs_{flowfs.mask_name}_{flowfs.stage_offset}mm_{flowfs.filter_name}_{num_modes}modes_{timestamp}{post_fix}.png", slopes=slopes, plot_crosstalk=True)
if args.plot:
    plt.show()

zero_points = np.mean(sweep_params[5::11], axis=1)[:9] #- np.array(biases)[:,None]
variation = zero_points - np.mean(zero_points, axis=0)
variation_rms = np.array([flowfs.rms(var, magellan_aperture) for var in variation])
background_noise = np.mean(variation_rms*flowfs.wavelength_wfs*1e9/(2*np.pi)) #nm
print('background variation (nm):', background_noise)

variation2 = np.std(variation, axis=0)
print('background variation (nm) from std:',flowfs.rms(variation2, magellan_aperture)*flowfs.wavelength_wfs*1e9/(2*np.pi)) #nm

os.makedirs(f'./results/{flowfs.mask_name}_{flowfs.stage_offset}mm_{flowfs.filter_name}_{num_modes}modes_{timestamp}{post_fix}', exist_ok=True)
np.save(f'./results/{flowfs.mask_name}_{flowfs.stage_offset}mm_{flowfs.filter_name}_{num_modes}modes_{timestamp}{post_fix}/sweep_params_linear.npy', sweep_params_linear)
np.save(f'./results/{flowfs.mask_name}_{flowfs.stage_offset}mm_{flowfs.filter_name}_{num_modes}modes_{timestamp}{post_fix}/sweep_params.npy', sweep_params)
np.save(f'./results/{flowfs.mask_name}_{flowfs.stage_offset}mm_{flowfs.filter_name}_{num_modes}modes_{timestamp}{post_fix}/acts.npy', acts)
np.save(f'./results/{flowfs.mask_name}_{flowfs.stage_offset}mm_{flowfs.filter_name}_{num_modes}modes_{timestamp}{post_fix}/biases.npy', biases)
np.save(f'./results/{flowfs.mask_name}_{flowfs.stage_offset}mm_{flowfs.filter_name}_{num_modes}modes_{timestamp}{post_fix}/slopes.npy', slopes)
np.save(f'./results/{flowfs.mask_name}_{flowfs.stage_offset}mm_{flowfs.filter_name}_{num_modes}modes_{timestamp}{post_fix}/background.npy', background_noise)


flowfs.zernike2dm_slopes = np.array(slopes)
flowfs.biases = np.array(biases)
flowfs.save_config(f'configs/flowfs_{flowfs.mask_name}_{flowfs.stage_offset}mm_{flowfs.filter_name}_{num_modes}modes_{timestamp}{post_fix}.json')
print(f'configs/flowfs_{flowfs.mask_name}_{flowfs.stage_offset}mm_{flowfs.filter_name}_{num_modes}modes_{timestamp}{post_fix}.json')