from reservoir_mass_balance import Reservoir, generate_streamflow
import numpy as np
import pytest
np.random.seed(0)

def test_calculate_area():
    """Test reservoir area calculation, mass balance, and exceptions.
    """
    n_weeks = 522
    storage_area_curve = np.array([[0, 500, 800, 1000], [0, 400, 600, 900]])

    evaporation = [0.5 / 10] * 3
    demands = [0.5 * 400] * 3
    inflows = [0.5 * 30, 0.5 * 30, 400] 
    reservoir1 = Reservoir(storage_area_curve, evaporation,
                           inflows, demands)

    # Test specific values of storages and areas
    assert reservoir1.calculate_area(500) == 400
    assert reservoir1.calculate_area(650) == 500
    assert reservoir1.calculate_area(1000) == 900

    # Test mass balance
    inflow = 100
    outlfow_test, _ = reservoir1.mass_balance(inflow, 1)
    assert outlfow_test == 0
    assert reservoir1.get_stored_volume_series()[1] == 870
    outlfow_test, _ = reservoir1.mass_balance(inflow, 2)
    assert outlfow_test == 134.75
    assert reservoir1.get_stored_volume_series()[2] == 1000

    # Test if exceptions are properly raised
    with pytest.raises(ValueError):
        reservoir1.calculate_area(-10)
        reservoir1.calculate_area(1e6)
        reservoir1.mass_balance(0, 0)


def test_generate_streamflow():
    n_weeks = 100000
    log_mu = 7.8
    log_sigma = 0.5
    sin_amplitude = 1.
    streamflows = generate_streamflow(n_weeks, sin_amplitude,
                                      log_mu, log_sigma)

    whitened_log_mu = (np.log(streamflows[range(0, n_weeks, 52)]) -
                       log_mu) / log_sigma

    # Test if whitened mean in log space is 0 +- 0.5
    assert np.mean(whitened_log_mu) == pytest.approx(0., abs=0.05)
