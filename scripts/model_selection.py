import tensorflow_hub
import tensorflow_text as text
import constants.trainer_constants
from constants.model_choices import map_name_to_handle, map_model_to_preprocess

SELECTED_MODEL = constants.trainer_constants.SELECTED_MODEL


def get_encoder_and_preprocessor():
    tfhub_handle_encoder = map_name_to_handle[SELECTED_MODEL]
    tfhub_handle_preprocessor = map_model_to_preprocess[SELECTED_MODEL]
    bert_preprocess_model = tensorflow_hub.KerasLayer(tfhub_handle_preprocessor, name='Preprocessor')
    bert_model = tensorflow_hub.KerasLayer(tfhub_handle_encoder, trainable=True, name=f'{SELECTED_MODEL}-Encoder')
    return bert_model, bert_preprocess_model
