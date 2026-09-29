from flowfs import *
from hcipy import *
import pytest
import numpy as np
import json


config_files = [
    'configs/test_config_updated.json',

]

@pytest.mark.parametrize("config_path", config_files)
def test_load_config(config_path):
    """Test the load_config function of the FLOWFS class."""
    flowfs = FLOWFS()
    # config_path = 'configs/test_config_updated.json'

    with open(config_path, 'r') as f:
        config = json.load(f)

    flowfs.load_config(config_path)

    # Check if the attributes are loaded correctly
    # assert np.allclose(flowfs.start_theta, np.array(config['start_theta']))
    assert flowfs._projection == tuple(config['projection']), "Projection does not match"
    assert flowfs.f_number == config['F_number'], "F_number does not match"
    
    # Check non-telecentricity magnification
    params = config['M_non_telecentricity_params']
    stage_offset = config['stage_offset']  
    expected_m_non_tele = params[0] * stage_offset + params[1]
    assert np.isclose(flowfs.M_non_telecentricity, expected_m_non_tele), "M_non_telecentricity does not match expected value."

    # Check final magnification
    expected_mag = config['M_camera'] * expected_m_non_tele
    assert flowfs.mag == flowfs.f_camera/flowfs.f, "Magnification does not match f_camera/f"
    assert np.isclose(flowfs.mag, expected_mag), f"Final magnification does not match. {flowfs.mag} vs {expected_mag}"

    # Check if theta is updated
    assert np.allclose(flowfs.theta, np.array(config['start_theta'])), "Theta does not match start_theta from config."

    # Check if fitted_parameters are all True
    assert np.all(flowfs.fitted_parameters)

def test_save_and_load_config():
    """Test saving and then loading a config results in the same state."""
    # Setup first object and load base config
    flowfs1 = FLOWFS()
    base_config_path = 'configs/test_config.json'
    flowfs1.load_config(base_config_path)

    # Save its config
    temp_config_path = 'tests/temp_config.json'
    flowfs1.save_config(temp_config_path)

    # Setup second object and load the saved config
    flowfs2 = FLOWFS()
    flowfs2.load_config(temp_config_path)

    # Compare attributes
    assert np.allclose(flowfs1.theta, flowfs2.theta)
    assert flowfs1._projection == flowfs2._projection
    assert np.isclose(flowfs1.f_number, flowfs2.f_number)
    assert np.isclose(flowfs1.mag, flowfs2.mag)
    assert np.isclose(flowfs1.M_non_telecentricity, flowfs2.M_non_telecentricity)
    assert np.allclose(flowfs1.M_non_telecentricity_params, flowfs2.M_non_telecentricity_params)

    # Clean up the temporary file
    import os
    os.remove(temp_config_path)