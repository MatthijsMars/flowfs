import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1' # Vervang '4' door het gewenste aantal threads

import argparse
import time                
import re
import pickle
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from hcipy import *
from flowfs import *

from magaox.deformable_mirror import *
from magaox.camera import XCam
from magpyx.utils import ImageStream


parser = argparse.ArgumentParser(
                    prog='flowfs_take_ref.py',
                    description='Takes reference images of FLOWFS, calibrates the optical model and writes the calibration files to cacao and disk.',
                    )
parser.add_argument('n', '--num_calibration_images', type=int, default=64, help='Number of calibration images to take')
parser.add_argument('--timeout', type=float, default=1e-2, help='Time to wait between images (s)')
parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for parallel processing')
parser.add_argument('--mode', type=int, default=2, help='Calibration mode: 0: remove all upstream aberrations (according to model), 1: remove all but tip/tilt, 2: keep stable around current point')
parser.add_argument('--update_cacao', action='store_true', help='Update cacao shared memory with new calibration files')
args = parser.parse_args()


class XDeformableMirror(DeformableMirror):
    def __init__(self, dm='woofer', channel=2, mode_basis_generator=None):
        ''' The MagAO-X python interface for the deformable mirrors.

        Parameters
        ----------
        dm : string
            The name of the dm that we are connecting to.
        channel : int
            The dm channel that is being used.
        mode_basis_generator : Field generator
            A function that calculates the modes for the DM. Default uses the full actuator basis.
        '''
        
        if dm == 'dmwoofer':
            self.dmindex = 0
            self.num_across = 11
            self.size = np.array([1, 1])
        elif dm == 'dmtweeter':
            self.dmindex = 1
            self.num_across = 50
            self.size = np.array([1, 1])
        elif dm == 'dmncpc':
            self.dmindex = 2
            self.num_across = 34
            self.size = np.array([1, np.sqrt(2.0)])
        else:
            self.dmindex = 0
            self.num_across = 11

        if mode_basis_generator is None:
            mode_basis_generator = lambda grid : ModeBasis(np.eye(grid.size), grid)
        
        # DM write channel
        self.dmwrite = ImageStream('dm0{:d}disp0{:d}'.format(self.dmindex, channel))
        
        # The DM modes
        self.grid = make_pupil_grid(self.num_across, self.size)
        modes = mode_basis_generator(self.grid)
        super().__init__(modes)

    def send(self, sleep=None):
        self.dmwrite.write(self.surface.shaped.astype(np.float32))        
        if sleep is not None:
            time.sleep(sleep)
    
    def reset(self):
        self.flatten()
        self.send()

class XZernikeMirror(XDeformableMirror):
    def __init__(self, starting_mode=2, number_of_modes=5, dm='woofer', channel=2, D=1):
        ''' The MagAO-X python interface for a Zernike-modes basis deformable mirror.
        Parameters
        ----------
        starting_mode : int
            The index of the first mode of the mode basis.
        number_of_modes : int
            The number of modes in the created mode basis.
        dm : string
            The name of the dm that we are connecting to.
        channel : int
            The dm channel that is being used.
        '''

        mode_basis_generator = lambda grid: make_zernike_basis(number_of_modes, D, grid, starting_mode=starting_mode)
        super().__init__(dm=dm, channel=channel, mode_basis_generator=mode_basis_generator)



flowfs = FLOWFS()
input_config = 'configs/flowfs_config_base.json'
save_dir = f'/opt/MagAO-X/calib/camflowfs/calib_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
os.makedirs(save_dir, exist_ok=True)

dm_flat = XZernikeMirror(starting_mode=2, number_of_modes=9, dm='dmncpc', channel=8)



for i in range(3):
    sweep_file = take_sweep_images(flowfs, save_dir=save_dir)
    output_config = os.path.join(save_dir, os.path.basename(sweep_file).replace('.pkl', '.json'))
    process_sweep(sweep_file, input_config, output_config=output_config, num_frames=5, on_sky=False, num_workers=32, regularisation_strength=0, num_modes=9, plot=False)
    injection_file = take_injection_data(save_dir=save_dir)
    flowfs.load_config(output_config)

    dm_flat.actuators[:] -= 1*flowfs.biases[:9] / flowfs.zernike2dm_slopes
    dm_flat.send()



flowfs = FLOWFS()
flowfs.load_config(output_config)

start_theta = flowfs.start_theta.copy()

magaox_mask = make_magaox_bump_mask()
magellan_aperture = evaluate_supersampled(magaox_mask, flowfs.pupil_grid, 6)
wavefront = Wavefront(magellan_aperture, flowfs.wavelength_wfs)
wavefront.total_power = 1.0

mode = 2
if mode == 0:
    # remove all upstream aberrations 
    # TODO should correct for relative tip/tilt
    flowfs.start_theta[9:] = start_theta[9:]
    flowfs.start_theta[:9] = 0
elif mode == 1:
    # remove everything but leave tip/tilt offset as is
    flowfs.start_theta[9:] = start_theta[9:]
    flowfs.start_theta[:9] = 0
    flowfs.start_theta[:2] = start_theta[:2]
elif mode == 2:
    # keep stable around current point
    flowfs.start_theta[:] = start_theta[:]


solver_ref_im, linear_solver, reconstruction_matrix = flowfs.get_reconstruction_matrix(wavefront, rcond=1e-3)


CMmodesWFS_path = os.path.join(save_dir, 'CMmodesWFS.fits')
wfsref_path = os.path.join(save_dir, 'wfsref.fits')
wfsmask_path = os.path.join(save_dir, 'wfsmask.fits')
config_path = os.path.join(save_dir, 'flowfs_config.json')

write_fits(
    (reconstruction_matrix.reshape(-1,32,32)/np.array(flowfs.zernike2dm_slopes).reshape(-1,1,1)).astype(np.float32), 
    CMmodesWFS_path
    )
write_fits(
    solver_ref_im.astype(np.float32), 
    wfsref_path
)
threshold = 70
write_fits(
    (solver_ref_im > np.percentile(solver_ref_im, threshold)).astype(int), 
    wfsmask_path
)


if args.update_cacao:
    ans = subprocess.call(["/usr/local/milk/bin/milk-FITS2shm", CMmodesWFS_path, "aol2_CMmodesWFS"])
    ans = subprocess.call(["/usr/local/milk/bin/milk-FITS2shm", wfsref_path, "aol2_wfsref"])
    ans = subprocess.call(["/usr/local/milk/bin/milk-FITS2shm", wfsref_path, "aol2_wfsrefc"])
    ans = subprocess.call(["/usr/local/milk/bin/milk-FITS2shm", wfsmask_path, "aol2_wfsmask"])
else:
    print('/usr/local/milk/bin/milk-FITS2shm {} aol2_CMmodesWFS'.format(CMmodesWFS_path))
    print('/usr/local/milk/bin/milk-FITS2shm {} aol2_wfsref'.format(wfsref_path))
    print('/usr/local/milk/bin/milk-FITS2shm {} aol2_wfsrefc'.format(wfsref_path))
    print('/usr/local/milk/bin/milk-FITS2shm {} aol2_wfsmask'.format(wfsmask_path)) 
