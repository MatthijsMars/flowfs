import pytest
import numpy as np
from hcipy import *
from flowfs import *



def test_gradients():

    flowfs = FLOWFS()
    flowfs.load_config('configs/test_config.json')
    magaox_mask = make_magaox_bump_mask()
    magellan_aperture = evaluate_supersampled(magaox_mask, flowfs.pupil_grid, 6)

    wavefront = Wavefront(magellan_aperture, flowfs.wavelength_wfs)

    start_theta = flowfs.start_theta.copy()
    # flowfs.blur = False
    # flowfs.fitted_parameters[-2] = False
    # flowfs._set_aberration(start_theta)

    im_ref = flowfs.forward(wavefront)

    new_theta = start_theta + 0.1 * np.random.randn(flowfs.num_parameters)
    new_theta[-2] = 0 # turn off the change in blur

    analytic_grad = flowfs.gradient(new_theta, wavefront, im_ref)
    finite_grad = []
    for i in range(flowfs.num_parameters):
        if i != flowfs.num_parameters - 2:
            d_theta = np.zeros(flowfs.num_parameters)
            d_theta[i] = 1e-6
            grad = flowfs.finite_gradient(
                new_theta,
                d_theta,
                wavefront,
                im_ref
            )

            finite_grad.append(grad)
        else:
            finite_grad.append(0)

    assert np.allclose(analytic_grad, finite_grad, atol=1e-3), "{analytic_grad} != {finite_grad}".format(
        analytic_grad=analytic_grad,
        finite_grad=finite_grad
    )

@pytest.mark.skip(reason="Blurring not working properly yet")
def test_gradients_with_blurring():

    flowfs = FLOWFS()
    magaox_mask = make_magaox_bump_mask()
    magellan_aperture = evaluate_supersampled(magaox_mask, flowfs.pupil_grid, 6)

    wavefront = Wavefront(magellan_aperture, flowfs.wavelength_wfs)

    start_theta = flowfs.start_theta.copy()
    flowfs.blur = True
    start_theta[-2] = 1
    flowfs._set_aberration(start_theta)

    im_ref = flowfs.forward(wavefront)

    new_theta = start_theta + 0.1 * np.random.randn(flowfs.num_parameters)

    analytic_grad = flowfs.gradient(new_theta, wavefront, im_ref)
    finite_grad = []
    for i in range(flowfs.num_parameters):
        d_theta = np.zeros(flowfs.num_parameters)
        d_theta[i] = 1e-6
        grad = flowfs.finite_gradient(
            new_theta,
            d_theta,
            wavefront,
            im_ref
        )[i]

        finite_grad.append(grad)


    assert np.allclose(analytic_grad, finite_grad, atol=1e-3), "{analytic_grad} != {finite_grad}".format(
        analytic_grad=analytic_grad,
        finite_grad=finite_grad
    )