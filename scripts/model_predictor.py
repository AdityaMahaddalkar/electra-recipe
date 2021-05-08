import numpy as np

from constants.path_variables import INPUT_FILE_PATHS
from scripts.data_preprocessor import get_test_data


def predict_test(model):
    test_data = get_test_data()

    test_x = test_data['Text'].values

    try:
        predictions = model.predict(test_x)
        create_submission(predictions)
    except Exception as e:
        print(e)


def create_submission(predictions):
    labels = predictions.argmax(axis=1) + 1

    np.savetxt(INPUT_FILE_PATHS['submission'], labels, delimiter='\n', fmt="%d")
