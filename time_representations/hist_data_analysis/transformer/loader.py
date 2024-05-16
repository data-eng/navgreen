import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import logging
import utils
import datetime
import astral, astral.sun

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def include_time_repr(df, params, dts, cors, uniqs, args):
    """
    Add time representations to a DataFrame.

    :param df: dataframe
    :param dts: list of str
    :param cors: list of str
    :param uniqs: list of str
    :param args: list of arguments
    :return: dataframe
    """
    for i, dtime in enumerate(dts):
        timestamps = df['DATETIME']
        tr_cor = TimeRepr(timestamps, dtime, args=args[i][0])
        tr_uniq = TimeRepr(timestamps, dtime, args=args[i][1])

        df[f'COR_{dtime.upper()}'] = getattr(tr_cor, cors[i])
        df[f'UNIQ_{dtime.upper()}'] = getattr(tr_uniq, uniqs[i])

        params["t"].extend([f'COR_{dtime.upper()}', f'UNIQ_{dtime.upper()}'])

    return df, params

class TimeRepr():
    def __init__(self, timestamps, dtime, args):
        """
        Initializes a time representation class.

        :param timestamps: pandas series
        :param dtime: datetime attribute (e.g., 'day', 'month', 'date')
        :param args: list of arguments for the functions
        """
        self.timestamps = timestamps
        self.dtime = dtime
        self.args = args

    @property
    def sine(self):
        """
        Calculate sine representation of timestamps.
        
        :return: pandas series
        """
        period, _, shift = self.args
        self.timestamps = self.timestamps.dt.__getattribute__(self.dtime)

        sine_result = np.sin(np.pi*(self.timestamps-shift)/period)
        sine_result += sine_result.iloc[1]/10

        return sine_result

    @property
    def cosine(self):
        """
        Calculate cosine representation of timestamps.
        
        :return: pandas series
        """
        period, _, shift = self.args
        self.timestamps = self.timestamps.dt.__getattribute__(self.dtime)

        cosine_result = np.cos(np.pi*(self.timestamps-shift)/period)
        cosine_result += cosine_result.iloc[1]/10

        return cosine_result
    
    @property
    def sawtooth(self):
        """
        Calculate sawtooth representation of timestamps.
        
        :return: pandas series
        """
        period, _, shift = self.args
        self.timestamps = self.timestamps.dt.__getattribute__(self.dtime)

        sawtooth_result = (self.timestamps-shift)/period
        sawtooth_result += sawtooth_result.iloc[1]/10

        return sawtooth_result
    
    @property
    def cond_sawtooth(self):
        """
        Calculate conditional sawtooth representation of timestamps.
        
        :return: numpy array
        """
        period, total, shift = self.args
        self.timestamps = self.timestamps.dt.__getattribute__(self.dtime)

        cond_sawtooth_result = np.where(self.timestamps <= period, 
                                       (self.timestamps-shift)/period, 
                                       (total-self.timestamps-shift)/period)

        cond_sawtooth_result += cond_sawtooth_result[1]/10

        return cond_sawtooth_result

    @property   
    def triangular_pulse(self):
        """
        Calculate triangular pulse representation of timestamps.
        
        :return: numpy array
        """
        pulse = []
        timestamps_dates = self.timestamps.dt.date
        
        for date in timestamps_dates.unique():
            d0 = pd.Timestamp(date)
            d1 = pd.Timestamp(date + datetime.timedelta(days=1))

            timestamps_one_day = self.timestamps.loc[(self.timestamps >= d0) & (self.timestamps < d1)]
            datetimes = timestamps_one_day.to_list()

            where = astral.LocationInfo("Athens", "Greece", "Europe/Athens", 37.995849, 23.814583)
            sun = astral.sun.sun(where.observer, datetimes[0], tzinfo=where.timezone)

            s0 = sun["sunrise"]
            s1 = sun["noon"]
            s2 = sun["sunset"]

            slope0 = 1/(s1-s0).total_seconds()
            slope1 = -1/(s2-s1).total_seconds()

            m1 = map( lambda t: t.replace(tzinfo=where.tzinfo), datetimes )
            m2 = map( lambda t: (0.01 if (t<s0) or (t>s2) else \
                                ((t-s0).total_seconds()*slope0 if t < s1 else \
                                (1+(t-s1).total_seconds()*slope1))) , m1 )
            
            pulse.extend(list(m2))

        pulse = np.array(pulse) + pulse[1]/10
        
        return pulse
    
    @property   
    def fixed_triangular_pulse(self):
        """
        Calculate triangular pulse representation of timestamps using fixed times.
        
        :return: numpy array
        """
        pulse = []
        
        for timestamp in self.timestamps:
            hour = timestamp.hour
            
            if hour < 7 or hour >= 19:
                pulse_value = 0.01
            elif hour >= 7 and hour < 13:
                pulse_value = (hour - 7) / 6
            else:
                pulse_value = (19 - hour) / 6
            
            pulse.append(pulse_value)
        
        pulse = np.array(pulse) + pulse[1]/10
        
        return pulse
    
    @property
    def linear(self):
        """
        Calculate linear representation of timestamps.
        
        :return: numpy array
        """
        total = (self.timestamps.iloc[-1] - self.timestamps.iloc[0]).total_seconds()
        line = list(map(lambda t: 1e-9 + (t - self.timestamps.iloc[0]).total_seconds()/total, self.timestamps))

        line = np.array(line) + line[1]/10
    
        return line

def load(path, parse_dates, bin, time_repr, normalize=True):
    """
    Loads and preprocesses data from a CSV file.

    :param path: path to the CSV file
    :param parse_dates: columns to parse as dates in the dataframe
    :param normalize: normalization flag
    :param bin: y_bin
    :param time_repr: tuple
    :return: dataframe
    """
    params = {"X": ["OUTDOOR_TEMP", "PYRANOMETER"],
              "t": [],
              "ignore": [] 
              }
    
    df = pd.read_csv(path, parse_dates=parse_dates, low_memory=False)
    df.sort_values(by='DATETIME', inplace=True)

    df = df[(df['DATETIME'] > '2022-11-10') & (df['DATETIME'] < '2023-09-26')]

    #logger.info("All data: {} rows".format(len(df)))

    empty_days = df.groupby(df['DATETIME'].dt.date).apply(lambda x: x.dropna(subset=params["X"], how='all').empty)
    df = df[~df['DATETIME'].dt.date.isin(empty_days[empty_days].index)]

    #logger.info("Number of empty days: %d", empty_days.sum())
    #logger.info("Number of empty data points: %d", 8 * empty_days.sum())
    #logger.info("Data after dropping NAN days: {} rows".format(len(df)))

    df, params = include_time_repr(df, params, *time_repr)

    if os.path.exists('transformer/stats.json'):
        stats = utils.load_json(filename='transformer/stats.json')
    else:
        stats = utils.get_stats(df, path='transformer/')

    occs = df[bin].value_counts().to_dict()
    freqs = {int(key): value / sum(occs.values()) for key, value in occs.items()}
    utils.save_json(data=freqs, filename=f'transformer/freqs_{bin}.json')

    inverse_occs = {int(key): 1 / value for key, value in occs.items()}
    weights = {key: value / sum(inverse_occs.values()) for key, value in inverse_occs.items()}
    
    if not os.path.exists(f'transformer/weights_{bin}.json'):
        utils.save_json(data=weights, filename=f'transformer/weights_{bin}.json')
    #else: print("Weights file already exists. Skipping saving!")

    if normalize:
        df = utils.normalize(df, stats, exclude=['DATETIME', 'COR_MONTH', 'COR_DAY', 'COR_HOUR', 'COR_DATE', 
                                                 'UNIQ_MONTH', 'UNIQ_DAY', 'UNIQ_HOUR', 'UNIQ_DATE', bin])

    nan_counts = df.isna().sum() / len(df) * 100
    #logger.info("NaN counts for columns in X: %s", nan_counts)

    return df, params

def prepare(df, phase, ignore):
    """
    Prepares the dataframe for training by filtering columns and saving to CSV.

    :param df: dataframe
    :param phase: str model phase (train or test)
    :return: dataframe
    """
    name =  "transformer/" + "df_" + phase + ".csv"

    for column, threshold in ignore:
        df = utils.filter(df, column=column, threshold=threshold) 

    df.set_index('DATETIME', inplace=True)
    df.to_csv(name)
    df = pd.read_csv(name, parse_dates=['DATETIME'], index_col='DATETIME')

    return df
    
class TSDataset(Dataset):
    def __init__(self, df, seq_len, X, t, y, per_day=False):
        """
        Initializes a time series dataset.

        :param df: dataframe
        :param seq_len: length of the input sequence
        :param X: input features names
        :param t: time-related features names
        :param y: target variables names
        :param per_day: boolean
        """
        self.seq_len = seq_len
        self.per_day = per_day

        y_nan = df[y].isna().any(axis=1)
        df.loc[y_nan, :] = float('nan')

        self.X = pd.concat([df[X], df[t]], axis=1)
        self.y = df[y]

    def __len__(self):
        """
        :return: number of sequences that can be created from dataset X
        """
        if not self.per_day:
            return self.max_seq_id + 1
        else:
            return self.num_seqs
    
    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        :param idx: index of the sample
        :return: tuple containing input features sequence, target variables sequence and their respective masks
        """
        if not self.per_day:
            start_idx = idx
        else:
            start_idx = idx * self.seq_len

        end_idx = start_idx + self.seq_len
    
        X, y = self.X.iloc[start_idx:end_idx].values, self.y.iloc[start_idx:end_idx].values

        mask_X, mask_y = pd.isnull(X).astype(int), pd.isnull(y).astype(int)

        X, y = torch.FloatTensor(X), torch.FloatTensor(y)
        mask_X, mask_y = torch.FloatTensor(mask_X), torch.FloatTensor(mask_y)

        X, y = X.masked_fill(mask_X == 1, -2), y.masked_fill(mask_y == 1, 0)

        seq_len = mask_X.size(0)
        mask_X_1d = torch.zeros(seq_len)
        mask_y_1d = torch.zeros(seq_len)

        for i in range(seq_len):
            if torch.any(mask_X[i] == 1):
                mask_X_1d[i] = 1
            if torch.any(mask_y[i] == 1):
                mask_y_1d[i] = 1

        return X, y, mask_X_1d, mask_y_1d
    
    @property
    def max_seq_id(self):
        return self.X.shape[0] - self.seq_len
    
    @property
    def num_seqs(self):
        return self.X.shape[0] // self.seq_len
    
def split(dataset, vperc=0.2):
    """
    Splits a dataset into training and validation sets.

    :param dataset: dataset
    :param vperc: percentage of data to allocate for validation
    :return: tuple containing training and validation datasets
    """
    ds_seqs = int(len(dataset))

    valid_seqs = int(vperc * ds_seqs)
    train_seqs = ds_seqs - valid_seqs

    return random_split(dataset, [train_seqs, valid_seqs])