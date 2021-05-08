import os.path

import tensorflow as tf

import constants.trainer_constants
from scripts.data_preprocessor import get_train_data, get_val_data
from scripts.model_builder import build_classifier_model

EPOCHS = constants.trainer_constants.EPOCHS
BATCH_SIZE = constants.trainer_constants.BATCH_SIZE


def compile_model(model):
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = [tf.metrics.CategoricalAccuracy(), tf.metrics.Precision()]

    optimizer = tf.keras.optimizers.Adadelta(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


def train_model():
    classifier_model = build_classifier_model()

    classifier_model = compile_model(classifier_model)

    train_data = get_train_data()

    val_data = get_val_data()

    train_x = train_data['Text'].values
    train_y = train_data.drop(['Text', 'Labels'], axis=1).values

    val_x = val_data['Text'].values
    val_y = val_data.drop(['Text', 'Labels'], axis=1).values

    callbacks = get_callbacks()

    history = classifier_model.fit(
        x=train_y,
        y=train_y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(val_x, val_y),
        callbacks=callbacks
    )

    return history, classifier_model


def get_callbacks():
    # tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(os.getcwd(), "logs"), write_images=True
    )

    # checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), "checkpoints", "cpkt")
    )

    # reduce LR on plateau
    lr_reducer_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, verbose=1
    )

    # early stopping
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_acc', min_delta=0.005, patience=3
    )

    return [tensorboard_callback, checkpoint_callback, lr_reducer_callback, early_stopping_callback]
