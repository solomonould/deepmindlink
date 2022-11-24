import os
import numpy as np
from numpy import array
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler
from model_file import n_timesteps, data_folder

# split a multivariate time series into short sequences
def split_sequences(sequences, n_steps):
    X, y = None, list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences.iloc[i:end_ix, :], sequences.iloc[end_ix-1, -1]
        
        if X is None:
            X = np.expand_dims(array(seq_x), axis=0)
        else:
            X = np.concatenate((X, np.expand_dims(array(seq_x), axis=0)), axis=0)
        y.append(seq_y)
    
    return X, array(y)


def merge_df(all_chunks, chunk):
    if all_chunks is None:
        return chunk
    else:
        return pd.concat([all_chunks, chunk])


def merge_list(dataset, data):
    if dataset is None:
        return data
    else:
        return np.concatenate((dataset, data), axis=0)


if __name__ == "__main__":

    # read raw data
    data = pd.read_csv(os.path.join(data_folder, 'Log1.csv'))
    column_names = ['Date', 'Temperature_C', 'pH', 'CO2_Injectionsper10min', 'LightIntensity', 'TankVolume', 'RelativeDensity']
    data = data[column_names]

    # filter out irrelevant records/events and split into chunks
    chunks = []
    start_idx = 0
    for idx in range(len(data)):
        if 'Event: ' in data.loc[idx, 'Date']:
            if 'Event: ' not in data.loc[idx-1, 'Date']:
                print(str(data.loc[start_idx:idx-1]))
                chunks.append(data.loc[start_idx:idx-1])
                start_idx = idx+1
            else:
                start_idx = idx+1
    chunks.append(data.loc[start_idx:len(data)-1])

    # remove redundant attributes
    for i in range(len(chunks)):
        chunks[i] = chunks[i].drop(['Date'], axis=1)
        chunks[i] = chunks[i].drop(['Temperature_C'], axis=1)
        chunks[i] = chunks[i].drop(['CO2_Injectionsper10min'], axis=1)
        chunks[i] = chunks[i].drop(['TankVolume'], axis=1)
        chunks[i] = chunks[i].astype({'pH':'float64', 'LightIntensity':'float64', 'RelativeDensity':'float64'})

    # define feature columns and target columns
    features_cols = ['pH', 'LightIntensity']
    target_cols = ['RelativeDensity']

    # normalize feature data in every chunk
        ## ideally, we should apply the scaler learned from training data to validation & test data
    scaler = MinMaxScaler()
    all_chunks = None
    for i in range(len(chunks)):
        all_chunks = merge_df(all_chunks, chunks[i])
    all_chunks[features_cols] = scaler.fit_transform(all_chunks[features_cols])
    for i in range(len(chunks)):
        chunks[i][features_cols] = scaler.transform(chunks[i][features_cols])

    # convert dataset into input/output
    data = list()
        # print('Number of chunks:', len(chunks))
        # print('Shape of chunk:', chunks[0].shape)
    for i in range(len(chunks)):
        X, y = split_sequences(chunks[i], n_timesteps)
        # print('Shape of X:', X.shape, 'Shape of Y:', y.shape)
        data.append((X, y))

    # split train data, validation data, and test data
    train_proportion, validation_proportion = 0.6, 0.2

    train_dataset_X, train_dataset_y = None, None
    validation_dataset_X, validation_dataset_y = None, None
    test_dataset_X, test_dataset_y = None, None

    test_dataset_sizes = list()

    for i in range(len(data)):
        X, y = data[i]
        
        train_size = int(train_proportion*len(X))
        validation_size = int(validation_proportion*len(X))
        test_size = len(X) - train_size - validation_size
        
        test_dataset_sizes.append(test_size)
        
        train_X, train_y = X[:train_size], y[:train_size]
        validation_X, validation_y = X[train_size:train_size+validation_size], y[train_size:train_size+validation_size]
        test_X, test_y = X[train_size+validation_size:], y[train_size+validation_size:]
        
        train_dataset_X, train_dataset_y = merge_list(train_dataset_X, train_X), merge_list(train_dataset_y, train_y)
        validation_dataset_X, validation_dataset_y = merge_list(validation_dataset_X, validation_X), merge_list(validation_dataset_y, validation_y)
        test_dataset_X, test_dataset_y = merge_list(test_dataset_X, test_X), merge_list(test_dataset_y, test_y)

    data_train = (train_dataset_X, train_dataset_y)
    data_valid = (validation_dataset_X, validation_dataset_y)
    data_test = (test_dataset_X, test_dataset_y)

    # print(len(data_train), len(data_valid), len(data_test))
    # print('train:', data_train[0].shape, data_train[1].shape)
    # print('valid:', data_valid[0].shape, data_valid[1].shape)
    # print('test:', data_test[0].shape, data_test[1].shape)

    # print(type(data_train), type(data_test[1]))
    # print('type of train:', type(data_train[0]))
    # print('type of valid:', type(data_valid[0]))
    # print('type of test:', type(data_test[0]))

    with open(os.path.join(data_folder, 'data_split.pkl'), 'wb') as fileObject:
        pkl.dump((data_train, data_valid, data_test), fileObject)
        print('Pre-processing completed!')