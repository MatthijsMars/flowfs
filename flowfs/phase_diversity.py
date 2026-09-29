import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"


import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
from scipy.optimize import minimize
from flowfs.visualisation import plot_difference
from flowfs.FLOWFS import FLOWFS
from tqdm.contrib.concurrent import process_map

def phase_retrieval(flowfs, start_theta, wavefront, im_ref, plot=True, fix_downstream_tip_tilt=False, fit_downstream=False, fit_f_camera=False, regularisation_strength=0.0, debug=False, ftol=1e-5, weights=1):
    flowfs.regularisation_strength = regularisation_strength
    flowfs.fitted_parameters[:] = True
    flowfs._set_aberration(start_theta)

    flowfs.fitted_parameters[:] = False
    flowfs.fitted_parameters[-1] = True
    flowfs.fitted_parameters[:2] = True
    if fix_downstream_tip_tilt:
        flowfs.fitted_parameters[flowfs.upstream_aberration_basis.num_modes: flowfs.upstream_aberration_basis.num_modes+2] = True
    else:
        flowfs.fitted_parameters[:2] = True

    res = minimize(flowfs.cost_cor, start_theta, args=(wavefront, im_ref, weights), jac=flowfs.gradient_cor, method='L-BFGS-B', options={'maxiter': 1000})
    if debug:
        print(res)

    start_theta[:] = res.x

    if fit_downstream:
        flowfs.fitted_parameters[:flowfs.upstream_aberration_basis.num_modes+flowfs.downstream_aberration_basis.num_modes] = True
    else:
        flowfs.fitted_parameters[:flowfs.upstream_aberration_basis.num_modes] = True
    if fit_f_camera:
        flowfs.fitted_parameters[-3] = True
    flowfs.fitted_parameters[-1] = True

    res = minimize(flowfs.cost_cor, start_theta, args=(wavefront, im_ref, weights), jac=flowfs.gradient_cor, method='L-BFGS-B', options={'maxiter': 1000})
    if debug:
        print(res)

    flowfs.fitted_parameters[:] = False
    flowfs.fitted_parameters[-1] = True

    res = minimize(flowfs.cost, res.x, args=(wavefront, im_ref), jac=flowfs.gradient, method='L-BFGS-B', options={'maxiter': 100})
    if debug:
        print(res)

    if plot:
        im = flowfs.forward(wavefront)
        plot_difference(im_ref, im, titles=['Observation', 'Model', 'Residual'])
        # plt.show()
        
    return res

def fun(flowfs, start_theta, wavefront, im_ref, plot=True, fix_downstream_tip_tilt=False, fit_downstream=False, fit_f_camera=False, regularisation_strength=0.0, debug=False, ftol=1e-5, weights=1):
    flowfs = FLOWFS()
    flowfs.load_config('/tmp/flowfs_config.json')

    res = phase_retrieval(flowfs, start_theta, wavefront, im, plot=plot, fix_downstream_tip_tilt=fix_downstream_tip_tilt, fit_downstream=fit_downstream, fit_f_camera=fit_f_camera, regularisation_strength=regularisation_strength, debug=debug, ftol=ftol, weights=weights)
    return res

from functools import partial
def multi_phase_retrieval(flowfs, start_theta, wavefront, im_refs, plot=True, fix_downstream_tip_tilt=False, fit_downstream=False, fit_f_camera=False, regularisation_strength=0.0, debug=False, ftol=1e-5, weights=1, num_workers=1):
    flowfs.save_config('/tmp/flowfs_config.json')

    fun_partial = partial(phase_retrieval, flowfs, start_theta, wavefront, plot=plot, fix_downstream_tip_tilt=fix_downstream_tip_tilt, fit_downstream=fit_downstream, fit_f_camera=fit_f_camera, regularisation_strength=regularisation_strength, debug=debug, ftol=ftol, weights=weights)

    fitted_params = []
    
    results = process_map(fun_partial, im_refs, max_workers=num_workers, chunksize=10)

    os.remove('/tmp/flowfs_config.json')
    return results
