from typing import Tuple

import pvlib
import pandas as pd


def get_pv_output(location: Tuple[float, float], panel_num: int, tracker: bool = False) -> pd.DataFrame:
    """
    calculate the output of a pv system
    :param location: the location of the system (lat, long)
    :param panel_num: number of panel in the system
    :param tracker: is the system using tracker
    :return:
    """
    if panel_num <= 0:
        raise ValueError("Number of panels should be positive")

    fixed_tilt = 25
    year_one = 1990
    times = pd.date_range(start=f'{year_one}-01-01 00:00', end=f'{year_one}-12-31 23:00', freq='h')

    # get tmy
    tmy = pvlib.iotools.get_pvgis_tmy(latitude=location[0], longitude=location[1], map_variables=True)[0]
    tmy.index = times

    # system definition
    location = pvlib.location.Location(latitude=location[0], longitude=location[1])
    module = pvlib.pvsystem.retrieve_sam('SandiaMod')['Canadian_Solar_CS5P_220M___2009_']
    inverter = pvlib.pvsystem.retrieve_sam('CECInverter')['ABB__PVI_3_0_OUTD_S_US__208V_']
    temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    # use different system for tracker or fixed
    if tracker:
        sat_mount = pvlib.pvsystem.SingleAxisTrackerMount()
        array = pvlib.pvsystem.Array(mount=sat_mount, module_parameters=module,
                                     temperature_model_parameters=temp_params)

        system = pvlib.pvsystem.PVSystem(arrays=[array], inverter_parameters=inverter, modules_per_string=1)
    else:
        system = pvlib.pvsystem.PVSystem(surface_tilt=fixed_tilt, surface_azimuth=180,
                                         module_parameters=module, inverter_parameters=inverter,
                                         temperature_model_parameters=temp_params,
                                         modules_per_string=1)

    # calculate the output (dc)
    model_chain = pvlib.modelchain.ModelChain(system, location)
    model_chain.run_model(tmy)
    # convert results from modelchain from W to kW
    return model_chain.results.ac / 1000 * panel_num
