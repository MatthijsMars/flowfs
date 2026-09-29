import pytest
import numpy as np
import matplotlib.pyplot as plt
from flowfs import *
from hcipy import *
from fpdf import FPDF
import datetime


configurations = [
    # config, im_ref
    ('configs/test_config_updated.json', "./tests/test_data/im_ref_14mm_lyot_sm.npy"),
    ('configs/test_config_knife_edge_z.json', "./tests/test_data/im_ref_14mm_knife_edge_z.npy")

]

@pytest.mark.parametrize("config_path, im_ref_path", configurations)
def test_convergence(config_path, im_ref_path):
    """Test the convergence of the FLOWFS algorithm."""
    flowfs = FLOWFS()
    flowfs.load_config(config_path)

    magaox_mask = make_magaox_bump_mask()
    magellan_aperture = evaluate_supersampled(magaox_mask, flowfs.pupil_grid, 6)
    wavefront = Wavefront(magellan_aperture, flowfs.wavelength_wfs)
    wavefront.total_power = 1.0

   
    im_ref = Field(np.load(im_ref_path), flowfs.binned_grid)

    start_theta = flowfs.start_theta.copy()
    start_theta[:-3] = 0 # start from no aberrations
    start_theta[flowfs.upstream_aberration_basis.num_modes + 2] =  18 # add a lot of defocus
    result = phase_retrieval(flowfs, start_theta, wavefront, im_ref, plot=False, fix_downstream_tip_tilt=True, fit_downstream=True, fit_f_camera=False, regularisation_strength=0.0, debug=False, ftol=1e-5)
    
    plot_difference(im_ref, flowfs.forward(wavefront), titles=['Observation', 'Model', 'Residual'])
    plt.savefig(f"./tests/test_results/test_{flowfs.mask_name}_{flowfs.filter_name}_{flowfs.stage_offset}.png")

    assert np.allclose(flowfs.start_theta, result.x, atol=1e-2), f"Theta did not converge properly: {flowfs.theta} vs {start_theta}"

    assert result.success, "Fitting did not converge successfully."
    assert result.fun < 2e-5, "Final cost function value is too high."



def test_create_report():
    """Create a pdf file with the generated results"""
 
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for config_path, im_ref_path in configurations:
        flowfs = FLOWFS()
        flowfs.load_config(config_path)

        im_result_path = f"./tests/test_results/test_{flowfs.mask_name}_{flowfs.filter_name}_{flowfs.stage_offset}.png"
        pdf.add_page()
        pdf.cell(0, 10, f"Mask Name: {flowfs.mask_name}", ln=True)
        pdf.cell(0, 10, f"Filter Name: {flowfs.filter_name}", ln=True)
        pdf.cell(0, 10, f"Stage Offset: {flowfs.stage_offset} mm", ln=True)
        pdf.image(im_result_path, x=10, y=40, w=180)

    # Append timestamp to the report filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf.output(f"./tests/test_results/report_{timestamp}.pdf")
