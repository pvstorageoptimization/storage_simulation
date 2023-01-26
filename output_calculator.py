import datetime
import time
from typing import Tuple
import numpy as np
import pandas as pd

import pv_output_calculator

# degradation of battery for each year
BAT_DEG_TABLE = (1.0, 0.9244, 0.8974, 0.8771, 0.8602, 0.8446, 0.8321, 0.8191, 0.8059, 0.7928, 0.7796, 0.7664, 0.7533,
                 0.7402, 0.7271, 0.7141, 0.7010, 0.6879, 0.6748, 0.6618, 0.6487, 0.6356, 0.6225, 0.6094, 0.5963, 0.5832)


class OutputCalculator:
    """
    Calculates the hourly output of the pv system and the storage system, and the hourly consumption from the grid
    """

    def __init__(self, num_of_years: int,
                 grid_size: int,
                 pv_output_file: str = None,
                 location: Tuple[float, float] = None,
                 panel_num: int = 0,
                 tracker: bool = False,
                 bess_self_consumption: float = 100,
                 bess_rte_mvmv: float = 0.87,
                 battery_dod: float = 0.9,
                 battery_soh: float = 0.939,
                 battery_block_size: float = 372.736,
                 battery_hours: float = 5,
                 bess_discharge_start_hour: int = 17,
                 bat_deg_table: tuple[float, ...] = BAT_DEG_TABLE,
                 fill_battery_from_grid: bool = False):
        """
        initialize the calculator with info on the system
        :param num_of_years: number of year to calculate the output for
        :param grid_size: the size of the grid connection (kwh)
        :param pv_output_file: a file name for hourly output of a pv system. Should contain 3 columns with names 'date',
               'fixed', 'tracker'. Columns should contain: date and hour, fixed pv output, tracker pv output (optional)
        :param location: location of the system (lat,long) (ignored if file is supplied)
        :param panel_num: number of panel in the pv system (ignored if file is supplied)
        :param tracker: does the pv system work with tracker or not (ignored if file is supplied)
        :param bess_self_consumption: hourly self consumption of the battery
        :param bess_rte_mvmv: a factor for how much power from power transmitted to battery is left after losses
        :param battery_dod: depth of charge of the battery
        :param battery_soh: state of health of the battery
        :param battery_block_size: size of each block of batteries (kwh)
        :param battery_hours: number of hours the battery get supply
        :param bess_discharge_start_hour: the first hour at which to release power from the battery
        :param bat_deg_table: a tuple containing the degradation level of the battery in each year
        :param fill_battery_from_grid: whether to buy power from grid to fill battery if pv is not enough
        """
        # times variables
        self.num_of_years = num_of_years

        # pv output variables
        self.grid_size = grid_size
        if pv_output_file:
            if not pv_output_file.endswith('.csv'):
                return
            self._initial_data = pd.read_csv(pv_output_file, index_col=0)
            self._initial_data = self._initial_data.drop(labels='fixed' if tracker else 'tracker', axis=1)
            self._initial_data.index = pd.to_datetime(self._initial_data.index)
            # TODO: check for read exception and correct format
        else:
            self._initial_data = pd.DataFrame(pv_output_calculator.get_pv_output(location, panel_num, tracker))
        self._initial_data.columns = ['pv_output']

        self.bess_self_consumption = bess_self_consumption
        self._pcs_power = (grid_size + bess_self_consumption) / bess_rte_mvmv
        self.battery_dod = battery_dod
        self.battery_soh = battery_soh
        self.battery_block_size = battery_block_size
        self.battery_hours = battery_hours
        self.bess_discharge_start_hour = bess_discharge_start_hour
        self.bat_deg_table = bat_deg_table
        self.fill_battery_from_grid = fill_battery_from_grid

        # calculation variables
        self._annual_deg = 0.0035
        self._charge_loss = 0.015
        self._discharge_loss = 0.015
        self._transmission_loss = 0.01

        # results lists
        self._results = None
        self._output = None
        self._purchased_from_grid = None

    # region Properties
    @property
    def num_of_years(self):
        return self._num_of_years

    @num_of_years.setter
    def num_of_years(self, value: int):
        if value <= 0:
            raise ValueError("Number of years should be positive")
        self._num_of_years = value

    @property
    def grid_size(self):
        return self._grid_size

    @grid_size.setter
    def grid_size(self, value: int):
        if value < 0:
            raise ValueError("Grid size should be positive")
        self._grid_size = value

    @property
    def bess_self_consumption(self):
        return self._bess_self_consumption

    @bess_self_consumption.setter
    def bess_self_consumption(self, value: float):
        if value < 0:
            raise ValueError("Battery self consumption should be positive")
        self._bess_self_consumption = value

    @property
    def pcs_power(self):
        return self._pcs_power

    def set_pcs_power(self, rte_mvmv: float):
        if not 0 < rte_mvmv < 1:
            raise ValueError("Battery rte mvmv should be between 0 and 1")
        self._pcs_power = (self._grid_size + self._bess_self_consumption) / rte_mvmv

    @property
    def battery_dod(self):
        return self._battery_dod

    @battery_dod.setter
    def battery_dod(self, value: float):
        if not 0 < value < 1:
            raise ValueError("Battery dod should be between 0 and 1")
        self._battery_dod = value

    @property
    def battery_soh(self):
        return self._battery_soh

    @battery_soh.setter
    def battery_soh(self, value: float):
        if not 0 < value < 1:
            raise ValueError("Battery soh should be between 0 and 1")
        self._battery_soh = value

    @property
    def battery_block_size(self):
        return self._battery_block_size

    @battery_block_size.setter
    def battery_block_size(self, value: float):
        if value <= 0:
            raise ValueError("Battery block size should be positive")
        self._battery_block_size = value

    @property
    def battery_bol(self):
        return self._battery_bol

    @property
    def battery_blocks(self):
        return self._battery_blocks

    @property
    def battery_hours(self):
        return self._battery_hours

    @battery_hours.setter
    def battery_hours(self, new_value: int):
        """
        changes the number of hours the battery supply (and variables effected by it)
        :param new_value: the new value for the hours supplied by the battery
        """
        if new_value < 0:
            raise ValueError("Battery hours should be non negative")
        self._battery_hours = new_value
        self._battery_bol = self._pcs_power * new_value
        self._battery_blocks = self._battery_bol / self._battery_dod / self._battery_soh / self._battery_block_size

    @property
    def bess_discharge_start_hour(self):
        return self._bess_discharge_start_hour

    @bess_discharge_start_hour.setter
    def bess_discharge_start_hour(self, value: int):
        if not 0 <= value <= 23:
            raise ValueError("Battery discharge start hour should be between 0 and 23 (inclusive)")
        self._bess_discharge_start_hour = value

    @property
    def bat_deg_table(self):
        return self._bat_deg_table

    @bat_deg_table.setter
    def bat_deg_table(self, value: tuple[float, ...]):
        if len(value) < self._num_of_years:
            raise ValueError("Battery deg table should have at least num_of_years entries")
        self._bat_deg_table = value

    @property
    def fill_battery_from_grid(self):
        return self._fill_battery_from_grid

    @fill_battery_from_grid.setter
    def fill_battery_from_grid(self, value: bool):
        self._fill_battery_from_grid = value

    @property
    def results(self):
        return self._results

    @property
    def output(self):
        return self._output

    @property
    def purchased_from_grid(self):
        return self._purchased_from_grid

    # endregion

    def _get_data(self, year):
        """
        create the basic data frame for the year (indexed by the date, with values of the pv system output
        :param year: the number of year in the simulation (first year is 0)
        """
        # in the first year Copy the range of years, PV and sunset/sunrise of first year from PV Worksheet
        if year == 0:
            self._df = self._initial_data.copy(deep=False)
        else:
            # calculate range of years, PV and sunset/sunrise for the year
            was_leap_year = (self._df.index[0].year % 400 == 0) or ((self._df.index[0].year % 100 != 0) and
                                                                    (self._df.index[0].year % 4 == 0))
            if was_leap_year:
                self._df = self._df.head(-24).copy(deep=False)
            else:
                self._df = self._df
            # if last year was leap years add 366 days and not 365
            self._df.index += datetime.timedelta(days=(366 if was_leap_year else 365))
            self._df = self._df * (1 - self._annual_deg)
            # checks if this is a leap year and add a day if so
            is_leap_year = (self._df.index[0].year % 400 == 0) or ((self._df.index[0].year % 100 != 0) and
                                                                   (self._df.index[0].year % 4 == 0))
            if is_leap_year:
                temp = self._df.tail(24)
                temp.index = self._df.index[-24:] + datetime.timedelta(days=1)
                self._df = pd.concat([self._df, temp])

        # adjust the battery capacity accounting for battery degradation
        self._current_battery_bol = self._battery_bol * self._bat_deg_table[year]

    def _calc_overflow(self):
        """
        calculate the hourly overflow of the pv output to bess and grid together
        """
        # calculate hourly pv production (including self consumption for pv and bess)
        self._df['pv_prod'] = self._df.iloc[:, 0] - self._bess_self_consumption
        # calculate overflow of power from pv and bess to grid
        temp = self._df['pv_prod'] - (self._grid_size - self._pcs_power) / (1 - self._transmission_loss)
        self._df['overflow'] = np.where(temp > 0, temp, 0)

    def _calc_pv_to_bess(self):
        """
        calculate the hourly transmission of power from pv to bess
        """
        # calculate available power from pv to bess (after reducing power from pv to grid, and limited by pcs
        # connection size)
        temp = np.minimum(self._df['pv_prod'] - self._grid_size / (1 - self._transmission_loss), self._pcs_power)
        prelim_pv_to_bess = np.where(temp > 0, temp, 0)
        # calculate the sum of power from pv to bess until each hour (starting from 6, sunrise)
        self._hour_since_six = self._df.index.hour - 6
        self._indices = np.arange(self._df.shape[0])
        reductions = np.column_stack((np.maximum(self._indices - self._hour_since_six, 0), self._indices)).ravel()
        temp_sums = np.add.reduceat(prelim_pv_to_bess, reductions)[::2]
        total_battery_trans = self._current_battery_bol / (1 - self._transmission_loss)
        # calculate the grid overflow that goes to the battery
        # if hourly pv to bess is bigger than remaining bess capacity only add the remaining capacity to bess
        self._df['pv2bess_pre'] = np.where(prelim_pv_to_bess > total_battery_trans - temp_sums,
                                           np.maximum(total_battery_trans - temp_sums, 0), prelim_pv_to_bess)
        # calculate loss due to battery size too small
        self._df['cap_loss'] = prelim_pv_to_bess - self._df['pv2bess_pre']

        # calculate the daily amount missing from the battery (daily underflow, including trans loss)
        temp = np.array(self._df['pv2bess_pre'])
        temp_sums = np.add.reduceat(temp, reductions)[::2]
        daily_underflow = np.where(self._df.index.hour == self._bess_discharge_start_hour,
                                   total_battery_trans - temp_sums, np.nan)
        daily_underflow = daily_underflow[~np.isnan(daily_underflow)]
        daily_underflow = np.repeat(daily_underflow, 24)

        # calculate corrected hourly pv to bess (accounting for the underflow in the hours close to the discharge hour)
        # get the index of the hour when starting to discharge
        discharge_hour_indices = np.repeat(np.where(self._df.index.hour == self._bess_discharge_start_hour), 24)
        # get the maximum extra pv that can be delivered to bess (accounting for available pv and remaining pcs
        # capacity)
        max_extra_pv_to_bess = np.array(np.minimum(np.maximum(self._df['pv_prod'], 0), self._pcs_power) -
                                        self._df['pv2bess_pre'])
        # sum the extra pv to bess to be added in the hours after each hour (and before start of discharge) to
        # compensate for the underflow
        reductions = np.column_stack((np.minimum(self._indices + 1, np.max(self._indices)), discharge_hour_indices)). \
            ravel()
        temp_sums = np.add.reduceat(max_extra_pv_to_bess, reductions)[::2]
        # calculate the added hourly pv to bess to compensate for underflow
        extra_pv_to_bess = np.where(self._df.index.hour < self._bess_discharge_start_hour,
                                    np.maximum(np.minimum(max_extra_pv_to_bess, daily_underflow - temp_sums), 0),
                                    0)
        # final pv to bess (including charge and trans losses)
        self._df['pv2bess'] = self._df['pv2bess_pre'] + extra_pv_to_bess

    def _calc_daily_initial_battery_soc(self):
        """
        calculate the battery soc before starting to discharge (including losses)
        """
        reductions = np.column_stack((np.maximum(self._indices - self._hour_since_six, 0), self._indices)).ravel()
        temp_sums = np.where(self._df.index.hour == self._bess_discharge_start_hour,
                             np.add.reduceat(np.array(self._df['pv2bess']), reductions)[::2], np.nan)
        initial_battery_soc = temp_sums[~np.isnan(temp_sums)]
        self.initial_battery_soc = np.repeat(initial_battery_soc, 24)

    def _calc_grid_to_bess(self):
        """
        calculate the power needed from the grid to fill the battery
        """
        max_rate = self._pcs_power / (1 - self._transmission_loss) - self._df['pv2bess']
        hours = self._df.index.hour
        missing_bess_power = self._current_battery_bol / (1 - self._transmission_loss) - self.initial_battery_soc
        self._df['grid2bess'] = np.where(self._bess_discharge_start_hour <= hours,
                                         np.maximum(np.minimum(max_rate,
                                                               missing_bess_power - (
                                                                       hours - self._bess_discharge_start_hour) *
                                                               max_rate * (1 - self._charge_loss) *
                                                               (1 - self._transmission_loss)),
                                                    0), 0)
        self.initial_battery_soc = np.full_like(self._df.iloc[:, 0], self._current_battery_bol)

    def _calc_power_to_grid(self):
        """
        calculate the hourly power transmitted to the grid from both pv and bess
        """
        # calculate hourly pv to grid
        self._df['pv2grid'] = np.minimum((self._df['pv_prod'] - self._df['pv2bess']) * (1 - self._transmission_loss),
                                         self._grid_size)

        # calculate hourly bess to grid + self consumption
        # maximum hourly from bess to grid given the hourly pv to grid
        max_rate = (self._grid_size - self._df['pv2grid']) / (1 - self._transmission_loss)
        # if battery soc is not minimal transmit the maximum that will not exceed the max_rate
        hours = self._df.index.hour
        self._df['bess2grid'] = np.where(self._bess_discharge_start_hour <= hours,
                                         np.maximum(np.minimum(max_rate,
                                                               self.initial_battery_soc
                                                               - (hours - self._bess_discharge_start_hour) *
                                                               max_rate),
                                                    0), 0) * (1 - self._transmission_loss) * (1 - self._discharge_loss)
        # calculate hourly power to grid (pv + bess)
        self._df['output'] = self._df['pv2grid'] + self._df['bess2grid']

    def run(self):
        """
        run the calculation and save hourly results into 'results':
            (pv_output - output of the pv system,
            pv_prod - pv output that can be used, without self consumption,
            overflow - pv overflow to grid and bess together,
            pv2bess_pre - power from pv to bess (before underflow),
            cap_loss - loss due to limit of battery capacity,
            pv2bess - power from pv to bess (after underflow),
            pv2grid - power from pv to grid,
            bess2grid - power from bess to grid,
            output - power form pv+bess to grid)
        and also save hourly output (last entry) to 'output'
        """
        self._results = []
        self._output = []
        self._purchased_from_grid = []
        for year in range(self._num_of_years):
            self._get_data(year)
            self._calc_overflow()
            self._calc_pv_to_bess()
            self._calc_daily_initial_battery_soc()
            if self._fill_battery_from_grid:
                self._calc_grid_to_bess()
            else:
                self._df['grid2bess'] = 0
            self._purchased_from_grid.append(self._df['grid2bess'])
            self._calc_power_to_grid()
            self._results.append(self._df)
            self._output.append(self._df['output'])

    def monthly_averages(self, year=0, stat='output'):
        """
        calculate and print the average in each hour for each month
        :param year: the year to calculate for
        :param stat: the stat to calculate
        """
        hours = np.tile(np.arange(0, 24), 365)
        months = np.repeat(np.arange(1, 13), np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) * 24)
        data = np.dstack([hours, months, self._results[year][stat]])[0]
        data = np.split(data[:, [0, 2]], np.unique(data[:, 1], return_index=True)[1][1:])
        data = [x[x[:, 0].argsort()] for x in data]
        data = [np.split(x[:, 1], np.unique(x[:, 0], return_index=True)[1][1:]) for x in data]
        data = [[round(y.mean()) for y in x] for x in data]
        np.set_printoptions(suppress=True)
        print(np.matrix(data))


if __name__ == '__main__':
    test = OutputCalculator(25, 5000, "test.csv")
    # test = OutputCalculator(1, 5000, location=(30.658611, 35.236667), panel_num=13648)
    start_time = time.time()
    test.run()
    print(f"calculation took: {time.time() - start_time} seconds")
    # test.monthly_averages(stat='pv_output')
