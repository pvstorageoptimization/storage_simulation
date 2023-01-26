import numpy as np
import numpy_financial as npf
import pandas as pd
import time

import output_calculator


class FinancialClaculator:
    """
    calculates finance for a pv system with storage
    """

    def __init__(self,
                 num_of_years: int,
                 land_size: int,
                 panel_num: int = 0,
                 panel_cost: int = 455,
                 battery_block_num: int = 0,
                 battery_block_cost: int = 3000,
                 usd_to_ils: float = 3.4,
                 rent_per_land_unit: int = 5000,
                 variable_opex: int = 1000000,
                 interest_rate: float = 0.05,
                 base_tariff: float = 0.14,
                 winter_low_factor: float = 1.04,
                 winter_high_factor: float = 3.91,
                 transition_low_factor: float = 1,
                 transition_high_factor: float = 1.2,
                 summer_low_factor: float = 1.22,
                 summer_high_factor: float = 6.28,
                 buy_from_grid_factor: float = 1.0):
        """
        initialize the calculator with info on the system
        :param num_of_years: number of year the system runs
        :param land_size: the size of the land
        :param panel_num: number of panels in the pv system
        :param panel_cost: cost of every panel ($)
        :param battery_block_num: number of battery block in the storage system
        :param battery_block_cost: cost of each battery block ($)
        :param rent_per_land_unit: cost of rent per land unit (shekel)
        :param variable_opex: variable operational cost
        :param interest_rate: the market interest rate
        :param base_tariff: basic tariff for power (multiplied by seasonal factor to get seasonal rate, shekel)
        :param winter_low_factor: winter low factor
        :param winter_high_factor: winter high factor
        :param transition_low_factor: transition season low factor
        :param transition_high_factor: transition season high factor
        :param summer_low_factor: summer low factor
        :param summer_high_factor: summer high factor
        """
        # expanses variables
        self.num_of_years = num_of_years
        self._rent_per_land_unit = rent_per_land_unit
        self.land_size = land_size
        self.usd_to_ils = usd_to_ils
        self._panel_cost = panel_cost
        self.panel_num = panel_num
        self._battery_block_cost = battery_block_cost
        self.battery_block_num = battery_block_num
        self.variable_opex = variable_opex
        self.interest_rate = interest_rate

        # revenues variables
        self._winter_months = [0, 1, 11]
        self._transition_months = [2, 3, 4, 9, 10]
        self._summer_months = [5, 6, 7, 8]
        # day of the week by datetime notation (monday is 0 and sunday is 6)
        self._week_days = [0, 1, 2, 3, 6]
        self._weekend_days = [4, 5]
        self._winter_low_hours = list(range(1, 17)) + [22, 23, 0]
        self._winter_high_hours = list(range(17, 22))
        self._transition_low_hours = self._winter_low_hours
        self._transition_high_hours = self._winter_high_hours
        self._summer_low_hours = list(range(1, 17)) + [23, 0]
        self._summer_high_hours = list(range(17, 23))
        self._winter_low = base_tariff * winter_low_factor
        self._winter_high_week = base_tariff * winter_high_factor
        self._transition_low = base_tariff * transition_low_factor
        self._transition_high_week = base_tariff * transition_high_factor
        self._summer_low = base_tariff * summer_low_factor
        self._summer_high_week = base_tariff * summer_high_factor
        self._winter_high_weekend = self._winter_high_week
        self._transition_high_weekend = self._transition_low
        self._summer_high_weekend = self._summer_low
        self.buy_from_grid_factor = buy_from_grid_factor

        self._income_details = None

        self._build_tariff_table()
        # save the last tariff matrix calculated
        self._tariff_matrix = None
        self._tariff_matrix_year = None

    def _reset_variables(self):
        self._income_details = None
        # save the last tariff matrix calculated
        self._tariff_matrix = None
        self._tariff_matrix_year = None

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
    def land_size(self):
        return self._land_size

    @land_size.setter
    def land_size(self, value: int):
        if value <= 0:
            raise ValueError("Land size should be positive")
        self._land_size = value
        self._rent_cost = value * self._rent_per_land_unit
        self._reset_variables()

    @property
    def usd_to_ils(self):
        return self._usd_to_ils

    @usd_to_ils.setter
    def usd_to_ils(self, value):
        if value < 0:
            raise ValueError("Dollar to shekel exchange should be non negative")
        self._usd_to_ils = value
        if hasattr(self, '_panel_num'):
            self.panel_num = self._panel_num
            self.battery_block_num = self._battery_block_num

    @property
    def panel_num(self):
        return self._panel_num

    @panel_num.setter
    def panel_num(self, value: int):
        if value <= 0:
            raise ValueError("Number of panels should be positive")
        self._panel_num = value
        self._total_panels_cost = value * self._panel_cost * self._usd_to_ils
        self._reset_variables()

    @property
    def total_panels_cost(self):
        return self._total_panels_cost

    @property
    def battery_block_num(self):
        return self._battery_block_num

    @battery_block_num.setter
    def battery_block_num(self, value):
        if value < 0:
            raise ValueError("Number of battery blocks should be non negative")
        self._battery_block_num = value
        self._battery_cost = value * self._battery_block_cost * self._usd_to_ils
        self._reset_variables()

    @property
    def battery_cost(self):
        return self._battery_cost

    @property
    def rent_cost(self):
        return self._rent_cost

    @property
    def variable_opex(self):
        return self._variable_opex

    @variable_opex.setter
    def variable_opex(self, value: int):
        if value < 0:
            raise ValueError("Variable opex should be non negative")
        self._variable_opex = value
        self._reset_variables()

    @property
    def interest_rate(self):
        return self._interest_rate

    @interest_rate.setter
    def interest_rate(self, value):
        if value <= 0:
            raise ValueError("Interest rate should be positive")
        self._interest_rate = value
        self._reset_variables()

    @property
    def buy_from_grid_factor(self):
        return self._buy_from_grid_factor

    @buy_from_grid_factor.setter
    def buy_from_grid_factor(self, value: float):
        if value < 0:
            raise ValueError("Factor for buying from grid should be positive")
        self._buy_from_grid_factor = value
        self._reset_variables()

    @property
    def income_details(self):
        return self._income_details
    # endregion

    def _build_tariff_table(self):
        """
        create a tariff table containing the tariff in each hour for each month
        """
        self._tariff_table = np.zeros((12, 7, 24))
        # winter tariffs
        self._tariff_table[np.ix_(self._winter_months, self._week_days, self._winter_low_hours)] = self._winter_low
        self._tariff_table[np.ix_(self._winter_months, self._weekend_days, self._winter_low_hours)] = self._winter_low
        self._tariff_table[np.ix_(self._winter_months, self._week_days, self._winter_high_hours)] = \
            self._winter_high_week
        self._tariff_table[np.ix_(self._winter_months, self._weekend_days, self._winter_high_hours)] = \
            self._winter_high_weekend
        # transition tariffs
        self._tariff_table[np.ix_(self._transition_months, self._week_days, self._transition_low_hours)] = \
            self._transition_low
        self._tariff_table[np.ix_(self._transition_months, self._weekend_days, self._transition_low_hours)] = \
            self._transition_low
        self._tariff_table[np.ix_(self._transition_months, self._week_days, self._transition_high_hours)] = \
            self._transition_high_week
        self._tariff_table[np.ix_(self._transition_months, self._weekend_days, self._transition_high_hours)] = \
            self._transition_high_weekend
        # summer tariffs
        self._tariff_table[np.ix_(self._summer_months, self._week_days, self._summer_low_hours)] = self._summer_low
        self._tariff_table[np.ix_(self._summer_months, self._weekend_days, self._summer_low_hours)] = self._summer_low
        self._tariff_table[np.ix_(self._summer_months, self._week_days, self._summer_high_hours)] = \
            self._summer_high_week
        self._tariff_table[np.ix_(self._summer_months, self._weekend_days, self._summer_high_hours)] = \
            self._summer_high_weekend

    def get_tariff_matrix(self, year):
        """
        create a matrix of hourly tariff in each day of the given year
        :param year: the year to calculate for
        :return: a numpy array with the tariffs
        """
        if self._tariff_matrix is not None and self._tariff_matrix_year == year:
            return self._tariff_matrix
        times = pd.date_range(start=f'{year}-01-01 00:00', end=f'{year}-12-31 23:00', freq='h', tz='Asia/Jerusalem')
        f = lambda x: self._tariff_table[x.month - 1, x.day_of_week, x.hour]
        self._tariff_matrix = f(times)
        self._tariff_matrix_year = year
        return self._tariff_matrix

    def get_power_sales(self, power_output, interest_rate: float = 0.0, purchased_from_grid=None):
        """
        calculate the yearly income according to the given power_output
        :param power_output: list of hourly output of the system for each year(list of pandas series)
        :param interest_rate: the interest rate in the market
        :param purchased_from_grid: list of hourly amount purchased from grid to fill battery (list of pandas series)
        :return: list of income per year
        """
        sales = []
        cpi_multi = 1
        self._income_details = []
        for year in range(self._num_of_years):
            tariff_matrix = self.get_tariff_matrix(power_output[year].index[0].year)
            temp = power_output[year] * tariff_matrix * cpi_multi
            temp = np.where(temp >= 0, temp, temp * self._buy_from_grid_factor)
            # add the payments for power from grid to battery
            if purchased_from_grid is not None:
                temp -= purchased_from_grid[year] * tariff_matrix * cpi_multi * self._buy_from_grid_factor
            sales.append(temp.sum())
            # save the matrices for each year
            self._income_details.append(temp)
            cpi_multi *= 1 + interest_rate
        return sales

    def get_expanses(self, interest_rate: float = 0.0):
        """
        calculate yearly expanses (capex+opex)
        :param interest_rate: the interest rate in the market
        :return: list of expanses per year
        """
        expanses = []
        cpi_multi = 1
        for year in range(self._num_of_years):
            if year == 0:
                expanses.append((self._total_panels_cost + self._battery_cost + self._rent_cost + self._variable_opex))
            else:
                expanses.append(self._rent_cost + self._variable_opex)
            expanses[year] *= cpi_multi
            cpi_multi *= 1 + interest_rate
        return expanses

    def get_irr(self, power_output, purchased_from_grid=None):
        """
        calculates the irr for the system
        :param power_output: list of hourly output of the system for each year(list of pandas series)
        :param purchased_from_grid: list of hourly amount purchased from grid to fill battery (list of pandas series)
        :return:
        """
        income = self.get_power_sales(power_output, self._interest_rate, purchased_from_grid)
        costs = self.get_expanses(self._interest_rate)
        revenues = [x - y for x, y in zip(income, costs)]
        irr = npf.irr(revenues)
        return irr if not np.isnan(irr) else 0

    def get_lcoe(self, power_output, purchased_from_grid):
        """
        calculate the lcoe for the system
        :param power_output: list of hourly output of the system for each year(list of pandas series)
        :param purchased_from_grid: list of hourly amount purchased from grid to fill battery (list of pandas series)
        :return:
        """
        yearly_construction_cost = [0] * self._num_of_years
        yearly_construction_cost[0] = self._total_panels_cost + self._battery_cost
        yearly_operational_cost = [self._rent_cost] * self._num_of_years
        yearly_variable_cost = [self._variable_opex] * self._num_of_years
        self.get_power_sales(power_output, self.interest_rate, purchased_from_grid)
        power_costs = [-x[x < 0].sum() for x in self._income_details]

        lcoe = sum([yearly_construction_cost[i] + yearly_operational_cost[i] + yearly_variable_cost[i] +
                    power_costs[i] / (1 + self._interest_rate) ** i for i in range(self._num_of_years)]) / \
               sum([power_output[i].sum() / (1 + self._interest_rate) ** i for i in range(self._num_of_years)])
        return lcoe


if __name__ == '__main__':
    output = output_calculator.OutputCalculator(25, 5000, "test.csv")
    output.run()
    test = FinancialClaculator(output.num_of_years, 97, panel_num=13648, battery_block_num=output.battery_blocks)
    start_time = time.time()
    print(test.get_irr(output.output, output.purchased_from_grid))
    print(test.get_lcoe(output.output, output.purchased_from_grid))
    print(f"calculation took: {(time.time() - start_time)} seconds")
