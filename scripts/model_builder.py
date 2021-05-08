import tensorflow as tf

from scripts import model_selection


def build_classifier_model():
    encoder_layer, preprocessing_layer = model_selection.get_encoder_and_preprocessor()

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    encoder_inputs = preprocessing_layer(text_input)
    outputs = encoder_layer(encoder_inputs)

    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(12, activation='softmax')(net)

    return tf.keras.Model(text_input, net)
