from magaox.deformable_mirror import *
from magaox.camera import XCam
from magpyx.utils import ImageStream

import re
import time
import tqdm
import pickle
from datetime import datetime
import numpy as np
import purepyindi2 as indi

from hcipy import *
from flowfs.utils import *


# redefine these as they don't account for sqrt(2) projection of dmncpc in the dev-branch code
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




def take_sweep_images(flowfs, save_dir=None):
    ''' Take a series of images while sweeping the knife-edge mask across the beam..
    '''
    indi_client = indi.client.IndiClient()
    indi_client.connect()

    indi_client.get_properties()
    time.sleep(1)
    current_filter = lambda s: re.findall(r"name='([a-z -]*)', _value=<SwitchState.ON", str(indi_client[s].elements()))[0]
    fwfpm = re.findall(r"name='([a-zA-Z -]*)', _value=<SwitchState.ON", str(indi_client['fwfpm.filterName'].elements()))[0]
    fwlowfs = re.findall(r"name='([a-zA-Z -]*)', _value=<SwitchState.ON", str(indi_client['fwlowfs.filterName'].elements()))[0]
    fwpupil = re.findall(r"name='([a-zA-Z -]*)', _value=<SwitchState.ON", str(indi_client['fwpupil.filterName'].elements()))[0]
    stage_offset = indi_client['stageflowfs.position.current']


    params =  []
    d = []

    num_modes = 9
    camflowfs = XCam('camflowfs')
    dm = XZernikeMirror(starting_mode=2, number_of_modes=num_modes, dm='dmncpc', channel=7)
    dm.flatten()
    dm.send()

    n_ims = 20
    timeout = 1e-3
    rms = np.linspace(-0.1, 0.1, 11)
    data = []

    for n in tqdm(range(n_ims)):
        for i in tqdm(range(num_modes)):
            p = []
            dm.actuators[:] = 0
            for j in rms:
                dm.actuators[i] = j
                dm.send(sleep=timeout)
        
                im_camflowfs = camflowfs.grab()
                im_ref = process_im(im_camflowfs, flowfs.binned_grid)
            
                d.append([stage_offset, np.copy(dm.actuators), np.copy(dm.surface), im_ref])
            # imshow_field(im_ref**.5, cmap='inferno')
            # res = phase_retrieval(flowfs, start_theta, wavefront, im_ref, plot=True, fix_downstream_tip_tilt=True, fit_downstream=True, fit_f_camera=False, regularisation_strength=0)
            # p.append(res.x)

            
        dm.flatten()
        dm.send()

    if save_dir:
        file_name = os.path.join(save_dir, f'flowfs_sweep_{fwfpm}_{stage_offset:.1f}mm_{fwlowfs}_{num_modes}modes_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.pkl')
        pickle.dump(d, open(file_name,  'wb'))
        return file_name

    return d


def take_injection_data(num_realisations=4000, rms_max=0.4, n_ims=20, timeout=1e-3, random_seed=295927, save_dir=None):
    indi_client = indi.client.IndiClient()
    indi_client.connect()

    indi_client.get_properties()
    time.sleep(1)
    current_filter = lambda s: re.findall(r"name='([a-z -]*)', _value=<SwitchState.ON", str(indi_client[s].elements()))[0]
    fwfpm = re.findall(r"name='([a-zA-Z -]*)', _value=<SwitchState.ON", str(indi_client['fwfpm.filterName'].elements()))[0]
    fwlowfs = re.findall(r"name='([a-zA-Z -]*)', _value=<SwitchState.ON", str(indi_client['fwlowfs.filterName'].elements()))[0]
    fwpupil = re.findall(r"name='([a-zA-Z -]*)', _value=<SwitchState.ON", str(indi_client['fwpupil.filterName'].elements()))[0]
    stage_offset = indi_client['stageflowfs.position.current']

    num_modes = 9
    camflowfs = XCam('camflowfs')
    dm = XZernikeMirror(starting_mode=2, number_of_modes=num_modes, dm='dmncpc', channel=7)
    dm.flatten()
    dm.send()

    data2 = []

    np.random.seed(random_seed)
    rms_realisations = [np.random.uniform(0,rms_max) for i in range(num_realisations)]

    injections = []

    for i in range(num_realisations):
        dm.random(1)
        dm.actuators[:] = dm.actuators * rms_realisations[i] / np.linalg.norm(dm.actuators)
        injections.append(np.copy(dm.actuators))

    for j in range(n_ims):
        for i in tqdm(range(num_realisations)):
            dm.actuators[:] = injections[i]
            dm.send(sleep=timeout)
                
            im = camflowfs.grab()
            # if i == 0:
            #     cx, cy = np.unravel_index(im.argmax(), im.shape)

            data2.append( [stage_offset, np.copy(dm.actuators), np.copy(dm.surface), im] )
            dm.flatten()


    dm.flatten()
    dm.send()

    if save_dir:
        filename = os.path.join(save_dir, f"flowfs_random_{fwfpm}_{stage_offset:.1f}mm_{fwlowfs}_{num_modes}modes_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pkl")
        pickle.dump(data2, open(filename,  'wb'))
        return filename

    return data2