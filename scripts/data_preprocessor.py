import pandas as pd

from constants.path_variables import INPUT_FILE_PATHS


def get_train_data():
    train_df = pd.read_csv(INPUT_FILE_PATHS['train.x'], sep="|", header=None, names=['Text'])
    train_labels = pd.read_csv(INPUT_FILE_PATHS['train.y'], sep="|", header=None, names=['Label'])

    assert train_df.shape[0] == train_labels.shape[0]

    train_df = pd.concat([train_df, train_labels], axis=1)

    train_df = get_one_hot_encoding(train_df)

    return train_df


def get_val_data():
    val_df = pd.read_csv(INPUT_FILE_PATHS['val.x'], sep="|", header=None, names=['Text'])
    val_labels = pd.read_csv(INPUT_FILE_PATHS['val.y'], sep="|", header=None, names=['Label'])

    assert val_df.shape[0] == val_labels.shape[0]

    val_df = pd.concat([val_df, val_labels], axis=1)

    val_df = get_one_hot_encoding(val_df)

    return val_df


def get_test_data():
    test_df = pd.read_csv(INPUT_FILE_PATHS['test.x'], sep="|", header=None, names=['Text'])

    return test_df


def get_one_hot_encoding(df, column='Label'):
    one_hot = pd.get_dummies(df[column])
    df = pd.concat([df, one_hot], axis=1)
    return df
