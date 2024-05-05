"""
Author: BilhaqAD
Date: 05/05/2024
Here's the tuner module.
Usage:
- FOR TUNER MODEL TO GET BEST HYPERPARAMETERS IN TRAINING MODEL
"""
from typing import Any, Dict, NamedTuple, Text
import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from keras import layers
from keras_tuner.engine import base_tuner
from transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name
)

NUM_TRIALS = 5

TunerFnResult = NamedTuple('TunerFnResult', [
    ('tuner', base_tuner.BaseTuner),
    ('fit_kwargs', Dict[Text, Any]),
])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_binary_accuracy',
    mode='max',
    verbose=1,
    patience=10,
)


def gzip_reader_fn(filenames):
    """
    Reads TFRecord files from the given list of filenames with GZIP compression.
    
    Parameters:
        filenames (list): List of file paths to TFRecord files.
    
    Returns:
        tf.data.TFRecordDataset: A dataset containing the TFRecord data.
    """
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn(file_pattern, tf_transform_output, batch_size=64):
    """
    Generates a dataset for training a model using the given file pattern,
    TF Transform output, and batch size.

    Args:
        file_pattern (str): The file pattern to match the input data files.
        tf_transform_output (tf.TransformOutput): The output of the TF Transform preprocessing step.
        batch_size (int, optional): The batch size for the generated dataset. Defaults to 64.

    Returns:
        tf.data.Dataset: The generated dataset for training the model.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset


def get_model_tuner(hyperparameters):
    """
	Generates a model tuner using the given hyperparameters.

	Parameters:
	- hyperparameters: An object containing the hyperparameters for the model tuner.

	Returns:
	- model: A compiled Keras model with the specified hyperparameters.

	The function generates a model tuner by creating a Keras model with the specified hyperparameters. 
    It starts by defining the number of hidden layers, 
    the number of units in the dense layers, the dropout rate, 
    and the learning rate. It then creates input features for 
    the categorical and numerical features. 
    The input features are concatenated and passed through a dense layer 
    with a ReLU activation function. Additional hidden layers are added
    with dropout regularization. The final layer is a dense 
    layer with a sigmoid activation function. The model is 
    compiled with the Adam optimizer, 
    binary cross-entropy loss, and binary accuracy metric. 
    The model summary is printed and the model is returned.

	Note: The function assumes that the CATEGORICAL_FEATURES and 
    NUMERICAL_FEATURES variables are defined elsewhere in the code.

	Example usage:
	```
	hyperparameters = Hyperparameters()
	model = get_model_tuner(hyperparameters)
	```
	"""

    num_hidden_layers = hyperparameters.Choice(
        'num_hidden_layers',
        values=[1, 2, 3],
    )

    dense_unit = hyperparameters.Int(
        'dense_unit',
        min_value=16,
        max_value=256,
        step=32,
    )

    dropout_rate = hyperparameters.Float(
        'dropout_rate',
        min_value=0.1,
        max_value=0.9,
        step=0.1,
    )

    learning_rate = hyperparameters.Choice(
        'learning_rate',
        values=[1e-2, 1e-3, 1e-4],
    )

    input_features = []

    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(
            layers.Input(shape=(dim + 1,), name=transformed_name(key))
        )

    for feature in NUMERICAL_FEATURES:
        input_features.append(
            layers.Input(shape=(1,), name=transformed_name(feature))
        )

    concatenate = layers.concatenate(input_features)
    deep = layers.Dense(dense_unit, activation=tf.nn.relu)(concatenate)

    for _ in range(num_hidden_layers):
        deep = layers.Dense(dense_unit, activation=tf.nn.relu)(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    outputs = layers.Dense(1, activation=tf.nn.sigmoid)(deep)

    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    model.summary()

    return model


def tuner_fn(fn_args):
    """
	Generates a tuner function for hyperparameter tuning.

	Parameters:
	- fn_args: An object containing the function arguments.

	Returns:
	- TunerFnResult: A named tuple containing the tuner and fit_kwargs for training the model.

	The function initializes a TFTransformOutput object based on the transform graph path in fn_args. 
	It then generates training and evaluation datasets using the input_fn function. 
	A RandomSearch tuner is created with the specified hypermodel, 
    objective, max_trials, directory, and project_name. 
	The function returns a TunerFnResult named tuple with the tuner 
    and fit_kwargs that include the training and validation data, 
	train and eval steps, and callbacks.

	Note: The function assumes the presence of NUM_TRIALS and early_stopping in the scope.

	Example usage:
	```
	tuner_result = tuner_fn(fn_args)
	```
	"""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(fn_args.train_files[0], tf_transform_output)
    eval_set = input_fn(fn_args.eval_files[0], tf_transform_output)

    tuner = kt.RandomSearch(
        hypermodel=get_model_tuner,
        objective=kt.Objective(
            'binary_accuracy',
            direction="max",
        ),
        max_trials=NUM_TRIALS,
        directory=fn_args.working_dir,
        project_name='kt_randomsearch',
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_set,
            'validation_data': eval_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps,
            'callbacks': [early_stopping],
        }
    )
