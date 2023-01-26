import pandas as pd
import pytz


if __name__ == '__main__':
    year = 2022

    times = pd.date_range(start=f'{year}-10-29 22:00', end=f'{year}-11-01 05:00', freq='h', tz='Asia/Jerusalem')

    f = lambda x: x.day

    print(f(times))
