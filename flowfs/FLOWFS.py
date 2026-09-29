import json
import os
import numpy as np
from warnings import warn
from scipy import ndimage
from hcipy import *
from flowfs.utils import ParameterList, flowfs_parameters
from functools import partial

class FLOWFS(OpticalElement):
    """Reflective focal plane mask wfs"""

    def __init__(self, wavelength=908e-9):
        self.wavelength_wfs = wavelength # m 
        telescope_diameter = 6.5 # m, magellan diameter
        self.mask_diameter = 8.604e-3 # /9.000e-3 * telescope_diameter # bump_mask diameter

        self._projection = (1, 1)
        self.regularisation_strength = 0.0

        num_pupil_pixels = 64
        pupil_grid_diameter = 2 * self.mask_diameter #TODO is this where the two comes from in spatial resolution?
        self.pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter) # pad by factor 2

        self.f_number = 69 * 9.000e-3/8.604e-3 # f-number is 69 for 9mm pupil, scale to bump mask size
        self.f = 621e-3 # focal length for focal plane
        self.spatial_resolution = self.f_number * self.wavelength_wfs # spatial resolution
        self.dm_oversize = 34/30 # partial illumination  of the DM 
        ## grids
        # focal_grid = make_focal_grid(q=8, num_airy=20, spatial_resolution=self.spatial_resolution, focal_length=self.f)

        # num_px = 64
        # self.px_per_lambda_D = 2.54*(self.wavelength_wfs/520e-9) * 2 #  designed at 2.54 px per lambda/D at 520nm
        # num_airy = num_px/self.px_per_lambda_D/2

        ## optical elements
        self._set_mask('lyot_sm')
        self.knife_edge_offset = np.array([0,0])

        # camera
        self.f_camera = 0.4*self.f #TODO should probably change this to a more sensible start value
        self.spatial_resolution_camera = 6.5e-6 # 6.5um pixels
        self.camera_grid = make_pupil_grid(64, 32*self.spatial_resolution_camera)
        self.binned_grid = make_pupil_grid(32, 32*self.spatial_resolution_camera) 
        self.prop_wfs_camera = FraunhoferPropagator(self.pupil_grid, self.camera_grid, focal_length=self.f_camera)

        # aberrations
        self.upstream_aberration_basis = make_zernike_basis(num_modes=9, D=self.mask_diameter*self.dm_oversize, grid=self.pupil_grid, starting_mode=2) # only tip-tilt

        self.pupil_aberration_basis = self.upstream_aberration_basis # deprecated

        self.downstream_aberration_basis = make_zernike_basis(num_modes=5, D=self.mask_diameter*4, grid=self.pupil_grid, starting_mode=2) # diameter *4 to include the full plane (not only pupil as there will be scattered light outside the pupil), messes up the units of the coefficients
        self.num_parameters = self.upstream_aberration_basis.num_modes + self.downstream_aberration_basis.num_modes + 3
        self.start_theta = np.zeros(self.num_parameters) # + blur kernel size and normalisation constant
        # self.start_theta[self.upstream_aberration_basis.num_modes +2] = 0 # defocus startvalue

        #TODO, should probably get rid of these default values        
        self.start_theta[-2] = 0 # blur kernel default value
        self.start_theta[-1] = 1 # normailsation default value
        # self.start_theta[-3] = 0.2*self.f # measured by illuminating mask (mask_size.ipynb)

        self.theta = np.copy(self.start_theta)
        self.fitted_parameters = np.ones(self.num_parameters, dtype=bool)
        # self.upstream_parameters = np.arange(self.fitted_parameters) < self.upstream_aberration_basis.num_modes
        # self.downstream_parameters = np.arange(self.fitted_parameters) >= self.upstream_aberration_basis.num_modes
        self.M_non_telecentricity = 1
        self.M_non_telecentricity_params = [0.0, 1.0]

        self.upstream_aberration = PhaseApodizer(self.upstream_aberration_basis.transformation_matrix @ self.theta[:self.upstream_aberration_basis.num_modes])
        self.downstream_aberration = PhaseApodizer(self.downstream_aberration_basis.transformation_matrix @ self.theta[self.upstream_aberration_basis.num_modes:-3])
        
        self._set_aberration(self.start_theta)

    def create_upstream_aberration(self, num_upstream_modes):
        old_num_modes = self.upstream_aberration_basis.num_modes
        self.upstream_aberration_basis = make_zernike_basis(num_upstream_modes, self.mask_diameter*self.dm_oversize, self.pupil_grid, starting_mode=2) # only tip-tilt
        
        self.num_parameters = self.upstream_aberration_basis.num_modes + self.downstream_aberration_basis.num_modes + 3
        self.fitted_parameters = np.ones(self.num_parameters, dtype=bool)

        new_start_theta = np.zeros(self.num_parameters)
        new_start_theta[:min(num_upstream_modes, old_num_modes)] = self.start_theta[:min(num_upstream_modes, old_num_modes)] # set old upstream modes
        new_start_theta[num_upstream_modes:] = self.start_theta[old_num_modes:] # set all downstream modes and other params

        self.start_theta = new_start_theta
        self.theta = np.copy(self.start_theta)
        self._set_aberration(self.start_theta)

    def create_downstream_aberration(self, num_downstream_modes):
        #diameter *4 to include the full plane (not only pupil as there will be scattered light outside the pupil), messes up the units of the coefficients
        old_num_modes = self.downstream_aberration_basis.num_modes
        self.downstream_aberration_basis = make_zernike_basis(num_downstream_modes, self.mask_diameter*4, self.pupil_grid, starting_mode=2)
        
        self.num_parameters = self.upstream_aberration_basis.num_modes + self.downstream_aberration_basis.num_modes + 3
        self.fitted_parameters = np.ones(self.num_parameters, dtype=bool)

        new_start_theta = np.zeros(self.num_parameters)
        #TODO this hardcoded -3 should be changed throughout
        new_start_theta[:self.upstream_aberration_basis.num_modes + min(old_num_modes, num_downstream_modes)] = self.start_theta[:self.upstream_aberration_basis.num_modes + min(old_num_modes, num_downstream_modes)] # set old upstream and downstream modes
        new_start_theta[-3:] = self.start_theta[-3:] # set other params

        self.start_theta = new_start_theta
        self.theta = np.copy(self.start_theta)
        self._set_aberration(self.start_theta)

    def _set_filter(self, filter_name='z'):
        """Set the filter to be used"""
        self.filter_name = filter_name
        if filter_name == 'z':
            self.wavelength_wfs = 908e-9 # m 
        elif filter_name == 'r':
            self.wavelength_wfs = 650e-9 # m 
        elif filter_name == 'g':
            self.wavelength_wfs = 520e-9 # m 
        else:
            warn(f"Filter {filter_name} not recognised, using z-band")
            self.wavelength_wfs = 908e-9 # m 

        self.spatial_resolution = self.f_number * self.wavelength_wfs # spatial resolution


    def _set_mask(self, mask_name='lyot_sm'):
        """Set the mask to be used"""
        self.mask_name = mask_name
        if mask_name == 'lyot_sm' or mask_name == 'lyot_lg':
            if mask_name == 'lyot_sm':
                radius_fpm = 272e-6/2 # 272um diameter
            else:
                radius_fpm = 453e-6/2 # 453um diameter
            
            coronograph_grid = make_pupil_grid(32, radius_fpm*2*1.2 )
            
            self.radius_fpm = radius_fpm
            def reflective_mask_generator(grid):
                return Field(
                        ((grid.x) ** 2 + (grid.y) ** 2 < radius_fpm**2), grid
                    ).astype("complex")
            self.rfpm = OccultedLyotCoronagraph(
                self.pupil_grid,
                evaluate_supersampled(reflective_mask_generator, coronograph_grid, 8), # supersample mask to avoid hard edges
                focal_plane_mask_grid=coronograph_grid,
                focal_length=self.f
            )
        elif mask_name == 'knifemaskZ':
            offset_basis = make_zernike_basis(2, self.mask_diameter, self.pupil_grid, starting_mode=2)
            offset = offset_basis.transformation_matrix @ self.knife_edge_offset
            self.rfpm = KnifeEdgeLyotCoronagraph(self.pupil_grid, direction='+y', apodizer=np.exp(1j*offset))
        elif mask_name == 'knifemask':
            offset_basis = make_zernike_basis(2, self.mask_diameter, self.pupil_grid, starting_mode=2)
            offset = offset_basis.transformation_matrix @ self.knife_edge_offset
            self.rfpm = KnifeEdgeLyotCoronagraph(self.pupil_grid, direction='-y', apodizer=np.exp(1j*offset))

    def _set_f_camera(self, f_camera):
        self.f_camera = f_camera
        self.mag = self.f_camera / self.f
        self.theta[-3] = self.mag * self.f
        self.start_theta[-3] = self.mag * self.f
        self.prop_wfs_camera = FraunhoferPropagator(self.pupil_grid, self.camera_grid, focal_length=self.f_camera)
        
    def _set_blur(self, b=1):
        self.blur = True
        r2 = (self.camera_grid.as_('polar').r / self.spatial_resolution)**2
        q_kernel = 2.54*(self.wavelength_wfs/520e-9)

        self.gauss = Field(np.exp(-b * r2) * b/np.pi * 1/q_kernel**2, self.camera_grid)
        self.gauss /= np.sum(self.gauss)

        self.inv_gauss = Field((r2 - 1.0/b) * self.gauss, self.camera_grid)


    def _set_aberration(self, theta):
        """Set the aberrations"""
        self.theta[self.fitted_parameters] = theta[self.fitted_parameters]

        self.upstream_aberration.phase = self.upstream_aberration_basis.transformation_matrix @ self.theta[:self.upstream_aberration_basis.num_modes]
        self.downstream_aberration.phase = self.downstream_aberration_basis.transformation_matrix @ self.theta[self.upstream_aberration_basis.num_modes:-3]

        self.pupil_aberration = self.upstream_aberration # deprecated
        self.lyot_aberration = self.downstream_aberration # deprecated

        #TODO, fix or remove the blur functionality
        if self.theta[-2] != 0:
            self._set_blur(self.theta[-2])
        else:
            self.blur = False
        
        if self.fitted_parameters[-3]:
            self._set_f_camera(self.theta[-3])

    #TODO fix or remove this function
    def print_parameters(self):
        self.theta_names = [
            "tip (pupil)",
            "tilt (pupil)",
            "tip (lyot)",
            "tilt (lyot)",
            "defocus (lyot)",
            "astigmatism 45 (lyot)",
            "astigmatism 0 (lyot)",
            "blur kernel size",
            "normalisation constant"
        ]
        for i, name in enumerate(self.theta_names):
            print(f"{name}: {self.theta[i]}")

    @staticmethod
    def _power(wavefront):
        return (wavefront.real**2 + wavefront.imag**2) * wavefront.grid.weights
    
    def forward(self, wavefront):
        """Propagate the wavefront through the system"""
        wavefront = self.upstream_aberration.forward(wavefront)
        wavefront = self.rfpm.forward(wavefront)
        wavefront = self.downstream_aberration.forward(wavefront)
        wavefront = self.prop_wfs_camera.forward(wavefront)

        if self.blur:
            im = self.convolve_field(self._power(wavefront), self.gauss) * self.theta[-1]
        else: 
            im = self._power(wavefront) * self.theta[-1]
        
        binned_im = self._bin(im)
        return binned_im
    
    def forward_E(self, wavefront):
        """Propagate the wavefront through the system and return the electric field at the camera"""
        wavefront = self.upstream_aberration.forward(wavefront)
        wavefront = self.rfpm.forward(wavefront)
        wavefront = self.downstream_aberration.forward(wavefront)
        wavefront = self.prop_wfs_camera.forward(wavefront)
        return wavefront 


    def backward(self, wavefront):
        """Propagate the wavefront back through the system"""
        wavefront = self.prop_wfs_camera.backward(wavefront)
        wavefront = self.downstream_aberration.backward(wavefront)
        wavefront = self.rfpm.backward(wavefront)
        wavefront = self.pupil_aberration.backward(wavefront)
        return wavefront

    @staticmethod
    def convolve_field(data, kernel):
        return Field(ndimage.convolve(data.shaped, kernel.shaped).ravel(), data.grid)

    @staticmethod
    def no_blur(wavefront, kernel=None):
        return wavefront

    @staticmethod
    def gradient_l2_norm(im, im_ref):
        return 2 * (im - im_ref)

    def cost(self, theta, wavefront, im_ref):
        self._set_aberration(theta)
        im_ref.grid = self.binned_grid
        im = self.forward(wavefront)
        return np.sum(np.square(im - im_ref))

    def _bin(self, im):
        bin_x, bin_y = 2, 2
        return subsample_field(im, bin_x, self.binned_grid, statistic='sum')
        im = im.shaped
        binned_im = im.reshape((im.shape[0]//bin_y, bin_y, im.shape[1]//bin_x, bin_x)).sum(axis=3).sum(axis=1)
        binned_im.grid = self.binned_grid
        return binned_im.flatten()

    def _unbin(self, im):
        bin_x, bin_y = 2, 2
        im = im.shaped
        im = im.repeat(bin_y, axis=0).repeat(bin_x, axis=1)
        im = im.flatten()
        im.grid = self.camera_grid
        return im

    def gradient(self, theta, wavefront, im_ref):
        #  Gradient of L-2 Norm
        self._set_aberration(theta)
        
        im_ref.grid = self.binned_grid

        wf = self.pupil_aberration(wavefront)
        wf = self.rfpm.forward(wf)
        wf_pre_lyot = wf.copy()
        wf = self.downstream_aberration.forward(wf)
        # wf = self.blur_kernel.forward(wf)
        wf = self.prop_wfs_camera(wf) 
        # im = wf.power  * self.theta[-1]
        im = self._power(wf)  * self.theta[-1]

        if self.blur:
            delta_psf_analytical = self.convolve_field(im, self.inv_gauss)
            blur_im = self.convolve_field(im, self.gauss)
            im = blur_im

        binned_im = self._bin(im)
        grad = 2 * (binned_im - im_ref)
        grad = self._unbin(grad)

        if self.blur:
            grad_blur = -1 * np.sum(grad * delta_psf_analytical)
        else:
            grad_blur = 0

        grad_normalisation =  np.sum(grad * np.conj(im / self.theta[-1]))  
        grad = grad * np.conj(self.theta[-1])  

        # gradient for the ncp aberrations
        if self.blur:
            grad = self.convolve_field(grad, np.conj(self.gauss)) * 2 * wf.electric_field # blur and intensity gradient
        else:
            grad = grad * 2 * wf.electric_field # intensity gradient
      
        grad = self.prop_wfs_camera.backward( Wavefront(grad, self.wavelength_wfs)) 
        grad = self.downstream_aberration.backward(grad)

        grad_lyot = grad.copy()
        grad_pupil = grad.copy()


        grad_lyot = grad_lyot.electric_field * np.conj(wf_pre_lyot.electric_field) * self.pupil_grid.weights   
        grad_lyot = np.imag(grad_lyot) #* np.pi / wavelength_wfs * 4
        grad_lyot =  1 * self.downstream_aberration_basis.transformation_matrix.T @ (grad_lyot) # + 2*regularisation * self.surface

        grad_pupil = self.rfpm.backward(grad_pupil)
        grad_pupil = self.upstream_aberration.backward(grad_pupil)
        grad_pupil = grad_pupil.electric_field * np.conj(wavefront.electric_field * self.pupil_grid.weights)
        grad_pupil = np.imag(grad_pupil)
        grad_pupil = 1 * self.upstream_aberration_basis.transformation_matrix.T @ grad_pupil

        if self.fitted_parameters[-3]:
            grad_f_camera = self.gradient_f_camera(theta[-3], wavefront, im_ref)
        else:
            grad_f_camera = 0


        total_grad = np.append(
            np.append(grad_pupil, grad_lyot),  
            np.append(grad_f_camera, (grad_blur, grad_normalisation))
            )
        
        return total_grad * self.fitted_parameters

    def rms(self, x, aperture=None):
        "Calculates the phase rms based on the current aberration basis and an optional aperture mask."
        phase = self.upstream_aberration_basis.transformation_matrix @ x[:self.upstream_aberration_basis.num_modes]
        
        if aperture is None:
            aperture_mask = (self.upstream_aberration_basis.transformation_matrix @ np.array([.3]*self.upstream_aberration_basis.num_modes)) != 0 # default to the region where modes act on
        if aperture is not None:
            phase = phase * aperture.flatten()
            aperture_mask = aperture.flatten() != 0
        if np.count_nonzero(phase) == 0:
            return 0.0
        return np.std(phase[aperture_mask]) #/ np.sqrt(self.upstream_aberration_basis.num_modes) #/ np.pi # convert to radian rms


    @staticmethod
    def cor_loss(x,y, weights=1):
        weights = np.asarray(weights)  # Ensure weights is an array
        return 1 - np.sum(weights * x * y)**2 / (np.sum(weights * x**2) * np.sum(weights * y**2)) 

    @staticmethod
    def cor_loss_gradient(x, y, weights=1):
        weights = np.asarray(weights)  # Ensure weights is an array
        A = np.sum(weights * x * y)    # Weighted sum of x * y
        B = np.sum(weights * x**2)     # Weighted sum of x^2
        C = np.sum(weights * y**2)     # Weighted sum of y^2

        grad = -2 * (A * weights * y) / (B * C) + 2 * (A**2 * weights * x) / (B**2 * C)
        return grad

    
    def cost_cor(self, theta, wavefront, im_ref, weights=1):
        self._set_aberration(theta)
        im = self.forward(wavefront)
        return self.cor_loss(im, im_ref, weights) + self.regularisation_strength * np.sum(self.theta[self.fitted_parameters]**2) # regularisation term


    def gradient_cor(self, theta, wavefront, im_ref, weights=1):
        #  Gradient of L-2 Norm
        self._set_aberration(theta)
        
        # im_ref.grid = self.camera_grid

        wf = self.upstream_aberration(wavefront)
        wf = self.rfpm.forward(wf)
        wf_pre_lyot = wf.copy()
        wf = self.downstream_aberration.forward(wf)
        # wf = self.blur_kernel.forward(wf)
        wf = self.prop_wfs_camera(wf) 
        # im = wf.power  * self.theta[-1]
        im = self._power(wf)  * self.theta[-1]
        bin_x, bin_y = 2, 2

        im = im.shaped
        im = im.reshape((im.shape[0]//bin_y, bin_y, im.shape[1]//bin_x, bin_x)).sum(axis=3).sum(axis=1) 
        im = im.flatten()
        im.grid = self.binned_grid

        grad_blur = 0
        grad = self.cor_loss_gradient(im, im_ref, weights) 

        grad_normalisation =  np.sum(grad * im)  * 7.242210312100214
        
        grad = grad.shaped
        grad = grad.repeat(bin_y, axis=0).repeat(bin_x, axis=1)
        grad = grad.flatten()

        grad = grad * np.conj(self.theta[-1])  #TODO should this be before or after the normalisation grad?

        # gradient for the ncp aberrations
        if self.blur:
            grad = self.convolve_field(grad, np.conj(self.gauss)) * 2 * wf.electric_field # blur and intensity gradient
        else:
            grad = grad * 2 * wf.electric_field # intensity gradient
        
        grad = self.prop_wfs_camera.backward( Wavefront(grad, self.wavelength_wfs)) 
        # grad = self.blur_kernel.backward(grad)
        grad = self.downstream_aberration.backward(grad)

        grad_lyot = grad.copy()
        grad_pupil = grad.copy()


        grad_lyot = grad_lyot.electric_field * np.conj(wf_pre_lyot.electric_field) * self.pupil_grid.weights   
        grad_lyot = np.imag(grad_lyot) #* np.pi / wavelength_wfs * 4
        grad_lyot =  1 * self.downstream_aberration_basis.transformation_matrix.T @ (grad_lyot) # + 2*regularisation * self.surface

        grad_pupil = self.rfpm.backward(grad_pupil)
        grad_pupil = self.upstream_aberration.backward(grad_pupil)
        grad_pupil = grad_pupil.electric_field * np.conj(wavefront.electric_field * self.pupil_grid.weights)
        grad_pupil = np.imag(grad_pupil)
        grad_pupil = 1 * self.upstream_aberration_basis.transformation_matrix.T @ grad_pupil

        grad_f_camera = 0
        if self.fitted_parameters[-3]:
            f_camera = self.theta[-3]
            dt = 1e-6 * f_camera
            if dt != 0:
                propagator = self.prop_wfs_camera
                try:
                    self.prop_wfs_camera = FraunhoferPropagator(
                        self.pupil_grid, self.camera_grid, focal_length=f_camera - dt)
                    c1 = self.cor_loss(self.forward(wavefront), im_ref, weights)
                    self.prop_wfs_camera = FraunhoferPropagator(
                        self.pupil_grid, self.camera_grid, focal_length=f_camera + dt)
                    c2 = self.cor_loss(self.forward(wavefront), im_ref, weights)
                    grad_f_camera = (c2 - c1) / (2 * dt)
                finally:
                    self.prop_wfs_camera = propagator


        total_grad = np.append(
            np.append(grad_pupil, grad_lyot),  
            np.append(grad_f_camera, (grad_blur, grad_normalisation))
            )
        
        return total_grad * self.fitted_parameters + self.regularisation_strength * 2 * theta * self.fitted_parameters


    def cost_f_camera(self, f_camera, wavefront, im_ref):
        self.theta[-3] = f_camera
        return self.cost_cor(self.theta, wavefront, im_ref)

    def gradient_f_camera(self, f_camera, wavefront, im_ref):
        if self.fitted_parameters[-3]:
            dt = 1e-6 * f_camera
            if dt == 0:
                return 0
            self.theta[-3] = f_camera - dt
            # self._set_f_camera(f_camera - dt)
            c1 = self.cost(self.theta, wavefront, im_ref)
            # self._set_f_camera(f_camera + dt)
            self.theta[-3] = f_camera + dt
            c2 = self.cost(self.theta, wavefront, im_ref)

            self.theta[-3] = f_camera
            return (c2 - c1) / (2 * dt)
        else:
            return 0

    def finite_gradient(self, theta, d_theta, wavefront, im_ref):
        return (self.cost(theta + d_theta/2, wavefront, im_ref) - self.cost(theta - d_theta/2, wavefront, im_ref)) / np.linalg.norm(d_theta)
    
    def projection(self, p_x, p_y):
        """
        Add a projection effect to the imaging system by scaling the camera grid by p_x and p_y in their respective directions. 
        The final grid is the `binned_grid` after binning the pixels which has the correct grid sizes that correspond to the actual camera.
        """
        self._projection = (p_x, p_y)
        self.camera_grid = make_pupil_grid(64, 32*self.spatial_resolution_camera)
        self.camera_grid.scale(self._projection) 
        self.prop_wfs_camera = FraunhoferPropagator(self.pupil_grid, self.camera_grid, focal_length=self.f_camera)

    def change_F_number(self, f_number):
        """Change the F-number of the system"""
        self.f_number = f_number
        self.f = self.f_number * self.mask_diameter
        if hasattr(self.rfpm, 'prop'):
            self.rfpm.prop.focal_length = self.f # Some coronagraphs are not using this propagator (e.g. knifemask)
        self.mag = self.f_camera / self.f

    def magnification(self, m):
        """Change the magnification of the system"""
        self.mag = m
        self.start_theta[-3] = self.mag * self.f
        self.theta[-3] =  self.mag * self.f
        self._set_f_camera(self.mag * self.f)

    def magnification_non_telecentricity(self, non_telecentricity_params, stage_offset):
        """Change the magnification of the system based on non-telecentricity parameters"""
        self.M_non_telecentricity_params = non_telecentricity_params
        def linear(x,a,b):
            return a*x + b
        self.M_non_telecentricity = linear(stage_offset, *non_telecentricity_params)

    def offset_stage(self, stage_offset):
        """Change the stage offset of the system"""
        self.stage_offset = stage_offset
        self.magnification_non_telecentricity(self.M_non_telecentricity_params, stage_offset)
        self.magnification(self.M_camera * self.M_non_telecentricity)


    def save_config(self, filename):
        """Save the configuration of the FLOWFS to a file"""

        config_data = {
            'start_theta': self.theta.tolist(),
            'projection': list(self._projection),
            'upstream_aberration': self.theta[:self.upstream_aberration_basis.num_modes].tolist(),
            'downstream_aberration': self.theta[self.upstream_aberration_basis.num_modes:-3].tolist(),
            'system_parameters': self.theta[-3:].tolist(),
            'M_camera': self.M_camera,
            'F_number': self.f_number,
            'M_non_telecentricity_params': self.M_non_telecentricity_params,
            'mask_name': self.mask_name,
            'filter_name': self.filter_name,
            'stage_offset': self.stage_offset,
            'zernike2dm_slopes': getattr(self, 'zernike2dm_slopes', []).tolist(),
            'biases': getattr(self, 'biases', []).tolist(),
            'git_commit_hash': getattr(self, 'git_commit_hash', ''),
        }
        if self.mask_name.startswith('knifemask'):
            config_data['knife_edge_offset'] = list(self.knife_edge_offset)
        with open(filename, 'w') as f:
            json.dump(config_data, f, indent=4)

    def copy(self):
        """Create a copy of the FLOWFS object"""
        new_flowfs = FLOWFS()
        self.save_config('/tmp/temp_flowfs_config.json')
        new_flowfs.load_config('/tmp/temp_flowfs_config.json')
        os.remove('/tmp/temp_flowfs_config.json')
        return new_flowfs

    def load_config(self, config_file):
        config = json.load(open(config_file))
        self._set_filter(config.get('filter_name', 'z')) # always set filter before mask
        self.knife_edge_offset = np.array(config.get('knife_edge_offset', [0,0]))
        self._set_mask(config.get('mask_name', 'lyot_sm'))
        print(self.knife_edge_offset, self.mask_name)
        self.stage_offset = config.get('stage_offset', 14)
        self.zernike2dm_slopes = np.array(config.get('zernike2dm_slopes', []))
        self.biases = np.array(config.get('biases', []))

        self.create_upstream_aberration(num_upstream_modes=len(config['upstream_aberration']))
        self.create_downstream_aberration(num_downstream_modes=len(config['downstream_aberration']))

        self.start_theta = np.concatenate((
            np.array(config['upstream_aberration']),
            np.array(config['downstream_aberration']),
            np.array(config['system_parameters'])
        ))
        self.theta = self.start_theta.copy()
        self.num_parameters = len(self.start_theta)
        self.fitted_parameters = np.ones(self.num_parameters, dtype=bool)
        self.M_camera = config['M_camera']
        # self.start_theta = np.array(config['start_theta'])
        self.projection(*config['projection'])
        self.magnification_non_telecentricity(config['M_non_telecentricity_params'], self.stage_offset) 
        # self.change_F_number(config['F_number'])
        self.magnification(self.M_camera * self.M_non_telecentricity ) 
        self._set_aberration(self.start_theta)

    def get_reconstruction_matrix(self, wf, probe_amp=0.1, rcond=1e-1):

        slopes = []
        norms = []
        num_rec_modes = self.upstream_aberration_basis.num_modes

        start_theta = self.start_theta.copy()
        
        if np.any(start_theta[:num_rec_modes] != 0):
            print("Warning: start_theta upstream aberrations are not zero")
            # start_theta[:num_rec_modes] = 0


        self.fitted_parameters[:] = True
        self._set_aberration(start_theta)
        self.fitted_parameters[:] = False
        def forward(x):
            return self.forward(x) /self.theta[-1] # undo normalization

        image_ref = forward(wf) #flowfs.forward(wf)

        aberration = np.copy(start_theta)
        self.fitted_parameters[:num_rec_modes] = True
        for ind in range(num_rec_modes):
            slope = 0
            # Probe the phase response
            for s in [1, -1]:
                aberration = np.copy(start_theta)
                amp = np.zeros((num_rec_modes,))
                amp[ind] = s * probe_amp  

                aberration[:num_rec_modes] += amp
                self._set_aberration(aberration)
                image = forward(wf) #flowfs.forward(wf)
                slope += s * (image-image_ref)/(2 * probe_amp)

            slopes.append(slope)

        slopes = ModeBasis(slopes)

        a = slopes.transformation_matrix.T/np.sqrt(image_ref.reshape(1, -1))

        reconstruction_matrix = inverse_tikhonov(slopes.transformation_matrix, rcond=rcond, svd=None)

        def linear_solve(reconstruction_matrix, image_ref, image):
            return reconstruction_matrix.dot(image - image_ref)

        return image_ref, partial(linear_solve, reconstruction_matrix, image_ref), reconstruction_matrix