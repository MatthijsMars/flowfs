import numpy as np
from hcipy import make_obstructed_circular_aperture, make_circular_aperture, make_spider_infinite

__all__ = [
    'make_magaox_bump_mask',
]

def make_magaox_bump_mask(normalized=False, with_spiders=True):
    '''Make the Magellan bump mask.

    Parameters
    ----------
    normalized : boolean
        If this is True, the outer diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 6.5 meters.
    with_spiders: boolean
        If this is False, the spiders will be left out.

    Returns
    -------
    Field generator
        The Magellan aperture.
    '''

    magnification_factor = 1 #6.5/9e-3 # Mag factor to scale 9 mm bump mask up to 6.5 m pupil diameter
    mask_inner = 2.79e-3 * magnification_factor # meter
    mask_outer = 8.604e-3 * magnification_factor # meter

    bump_mask_diameter = 0.5742e-3 * magnification_factor

    bump_mask_pos = np.array([2.853e-3, -0.6705e-3]) * magnification_factor

    radius = np.hypot(bump_mask_pos[0], bump_mask_pos[1])
    theta = np.arctan2(bump_mask_pos[1], bump_mask_pos[0]) - np.deg2rad(25.2) #+ np.pi/2 # Adjusted bump angle to better center it on spider
    bump_mask_pos = np.array([radius * np.cos(theta), radius * np.sin(theta)]) * magnification_factor

    pupil_diameter = 6.5  # meter
    spider_width1 = 0.1917e-3 * magnification_factor  # meter 
    spider_width2 = 0.1917e-3  * magnification_factor  # meter
    central_obscuration_ratio = mask_inner / mask_outer 
    spider_offset = np.array([0.0, 0.34])/6.5*9e-3 *magnification_factor  # meter 

    if normalized:
        spider_width1 /= pupil_diameter
        spider_width2 /= pupil_diameter
        spider_offset /= pupil_diameter
        bump_mask_pos /= pupil_diameter
        bump_mask_diameter /= pupil_diameter
        pupil_diameter = 1.0

    obstructed_aperture = make_obstructed_circular_aperture(mask_outer, central_obscuration_ratio)
    bump_mask = make_circular_aperture(bump_mask_diameter, center=bump_mask_pos)  # Generate bump cover for the MagAO-X DM
    
    if not with_spiders:
        return obstructed_aperture
	
	# spider offsets corrections based on bumpMask fits file from J. Males.
    spider1 = make_spider_infinite(spider_offset - np.array([0, -0.0185])/6.5*9e-3 * magnification_factor, 45.0, spider_width1)
    spider2 = make_spider_infinite(-spider_offset - np.array([0, 0.020])/6.5*9e-3 * magnification_factor, -45.0, spider_width1)
    spider3 = make_spider_infinite(-spider_offset + np.array([0, -0.022])/6.5*9e-3 * magnification_factor, 45.0 + 180.0, spider_width2)
    spider4 = make_spider_infinite(spider_offset + np.array([0, 0.023])/6.5*9e-3 * magnification_factor, -45.0 + 180.0, spider_width2)

    def func(grid):
        return obstructed_aperture(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid) * (1 - bump_mask(grid))
    
    return func