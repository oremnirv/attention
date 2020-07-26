import pandas as pd
import numpy as np


def climate_data_to_model_input(path, n_rows=10000):
    '''

    '''
    df = pd.read_csv(path)
    df['dt'] = (pd.to_datetime(df['time']).dt.year % pd.to_datetime(
        df['time']).dt.year[0]) + (pd.to_datetime(df['time']).dt.month / 12)
    df.drop('Unnamed: 0', inplace=True, axis=1)
    combined_array = np.zeros((n_rows * 3, 40))
    for row in range(0, n_rows * 3, 3):
        uni = list(df['number'].unique())
        num = np.random.choice(uni, 2, replace=False)
        df_1 = df[df['number'] == num[0]]
        df_2 = df[df['number'] == num[1]]

        # pick differing sequence length
        seq_len = np.random.choice(np.arange(5, 20), 2)
        seq1_idx = sorted(list(np.random.choice(list(df_1.index)[:-21], seq_len[0])))

        df_2 = df_2[df_2['dt'] > max(df_1.loc[seq1_idx, 'dt'])]
        seq2_idx = sorted(list(np.random.choice(list(df_2.index), seq_len[1])))

        m = (seq_len[0] + seq_len[1])
        combined_array[row, :m] = np.concatenate(
            (df_1.loc[seq1_idx, 't2m'], df_2.loc[seq2_idx, 't2m']))
        combined_array[row + 1, :m] = np.concatenate(
            (df_1.loc[seq1_idx, 'dt'], df_2.loc[seq2_idx, 'dt']))
        combined_array[row + 2, :m] = np.concatenate(
            (df_1.loc[seq1_idx, 'number'], df_2.loc[seq2_idx, 'number']))
    temp = combined_array[0::3]
    t = combined_array[1::3]
    token = combined_array[2::3]
    return temp, t, token
