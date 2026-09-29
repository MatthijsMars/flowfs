import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc

from fixr import xrif2numpy
from hcipy import *
from flowfs.bump_mask import make_magaox_bump_mask


def load_file(i, folder):
    files = os.listdir(folder)
    fh = open(f'{folder}/{files[i]}', 'rb')
    data = xrif2numpy(fh)
    timings = xrif2numpy(fh)
    return data, timings

def load_batch(i, folder):
    files = os.listdir(folder)
    fh = open(f'{folder}/{files[i]}', 'rb')
    data = xrif2numpy(fh)
    timings = xrif2numpy(fh)

    for file in files[i:i+250]:
        # TODO this does not always result in 250 files
        if file.endswith('.xrif'):
            fh = open(f'{folder}/{file}', 'rb');
            data = np.append(data, xrif2numpy(fh), axis=0);
            # timings = np.append(timings, xrif2numpy(fh), axis=0);
    return data

def process_im(im, grid):
    im_ref = im.astype(float)

    background = np.min([ 
        np.median(im_ref[:2]),
        np.median(im_ref[-2:]),
        np.median(im_ref[:, :2]),
        np.median(im_ref[:, -2:])
    ])

    im_ref -= background
    im_ref /= np.sum(im_ref)
    im_ref = Field(im_ref.flatten(), grid)
    return im_ref

def loop(data):

    fig = plt.figure(figsize=(4,3))
    plt.subplot(1,1,1)
    plt.title(r'WFS')
    im1 = plt.imshow(data[0,0], vmin=data.min(), vmax=data.max())
    plt.colorbar()

    plt.close(fig)

    def animate(t):
        im1.set_data(data[t*80,0])

        fig.canvas.draw()
        plt.clf()
        return [im1]

    fps = 5
    s = 30
    num_time_steps = fps * s
    time_steps = np.arange(num_time_steps)
    anim = animation.FuncAnimation(fig, animate, time_steps, interval=160, blit=True)

    anim.save('data2.mp4', writer='ffmpeg', fps=fps)
    return anim 


class ParameterList():
    """
    A list of parameters for the FLOWFS simulator.
    """
    def __init__(self, l, dtype=None, keys=None):
        self._list = np.array(l, dtype=dtype)
        self._keys = keys

        if keys is not None:
            assert len(self._list) == len(self._keys), f"Number of keys must match number of parameters. {self._list, self._keys}"

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._list.__getitem__(self._keys.index(key))
        else:
            return self._list.__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self._list.__setitem__(self._keys.index(key), value)
        else:
            self._list.__setitem__(key, value)

    def __repr__(self):
        s = ""
        for i, key in enumerate(self._keys):
            s += f"{key}: {self._list[i]}\n"
        return s
        
flowfs_parameters = [
    "upstream tip",
    "upstream tilt",
    "upstream defocus",
    "upstream astigmatism 0",
    "upstream astigmatism 45",
    "upstream coma 0",
    "upstream coma 45",
    "upstream trefoil 0",
    "upstream trefoil 45",
    "downstream tip",
    "downstream tilt",
    "downstream defocus",
    "downstream astigmatism 0",
    "downstream astigmatism 45",
    "downstream focal length",
    "blur",
    "normalisation",
]


def semi_analytic_fisher_matrix(flowfs, normalise_modes=False):
    magaox_mask = make_magaox_bump_mask()
    magellan_aperture = evaluate_supersampled(magaox_mask, flowfs.pupil_grid, 8)

    wavefront = Wavefront(magellan_aperture, flowfs.wavelength_wfs)
    wavefront.total_power = 1
    E_0 = flowfs.forward_E(wavefront) 
    I_0 = flowfs._bin(np.abs(E_0.electric_field)**2 ) 
    dIs = []

    for i in range(flowfs.upstream_aberration_basis.num_modes ):
        wf = wavefront.copy()
        wf.total_power = 1
        wf.electric_field *= 1j * flowfs.upstream_aberration_basis.transformation_matrix[:,i] #
        if normalise_modes:
            wf.electric_field /= np.std(flowfs.upstream_aberration_basis.transformation_matrix[:,i][magellan_aperture>0])

        wf = flowfs.forward_E(wf)
        dIs.append(2 * flowfs._bin(np.real(np.conj(E_0.electric_field) * wf.electric_field)))

    FIM = np.zeros((flowfs.upstream_aberration_basis.num_modes, flowfs.upstream_aberration_basis.num_modes))
    for i in range(len(dIs)):
        for j in range(len(dIs)):
            FIM[i,j] = np.sum(dIs[i] * dIs[j] / (I_0) * E_0.grid.weights ) 
    
    return FIM

def numerical_fisher_matrix(flowfs, probe_amp=0.1):
    magaox_mask = make_magaox_bump_mask()
    magellan_aperture = evaluate_supersampled(magaox_mask, flowfs.pupil_grid, 8)
    wf = Wavefront(magellan_aperture, flowfs.wavelength_wfs)
    wf.total_power = 1

    forward = lambda x: flowfs.forward(x) /flowfs.theta[-1] # undo normalization

    slopes = []
    flowfs.start_theta[:-8] = 0
    start_theta = flowfs.start_theta.copy()

    flowfs.fitted_parameters[:] = True
    flowfs._set_aberration(start_theta)

    image_ref = forward(wf) #flowfs.forward(wf)

    aberration = np.copy(start_theta)
    num_rec_modes = flowfs.pupil_aberration_basis.num_modes
    flowfs.fitted_parameters[:num_rec_modes] = True
    for ind in range(num_rec_modes):
        slope = 0
        norm = 0
        # Probe the phase response
        for s in [1, -1]:
            aberration = np.copy(start_theta)
            amp = np.zeros((num_rec_modes,))
            amp[ind] = s * probe_amp  

            aberration[:num_rec_modes] += amp
            flowfs._set_aberration(aberration)
            image = forward(wf) #flowfs.forward(wf)
            slope += s * (image-image_ref)/(2 * probe_amp)

        slopes.append(slope)

    slopes = ModeBasis(slopes)

    a = slopes.transformation_matrix.T/np.sqrt(image_ref.reshape(1, -1))
    fisher_matrix = np.dot(a, a.T)
    return fisher_matrix


