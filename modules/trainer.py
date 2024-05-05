"""
Author: BilhaqAD
Date: 05/05/2024
Here's the trainer module.
Usage:
- FOR TRAINING THE MODEL
"""
import os
import tensorflow as tf
import tensorflow_transform as tft
from keras import layers
from transform import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    LABEL_KEY,
    transformed_name
)
from tuner import input_fn


def get_serve_tf_examples_fn(model, tf_transform_output):
    """
	Defines a function to serve TensorFlow examples. It takes a model and a TensorFlow 
    transform output as input.The function processes the serialized 
    TensorFlow examples, applies transformations, and returns the outputs.
	
	Args:
	    model: The TensorFlow model for processing the examples.
	    tf_transform_output: The TensorFlow Transform output for transforming features.

	Returns:
	    A dictionary containing the processed outputs.
	"""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
    ])
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(
            serialized_tf_examples,
            feature_spec
        )

        transformed_features = model.tft_layer(parsed_features)
        outputs = model(transformed_features)

        return {'outputs': outputs}

    return serve_tf_examples_fn


def get_model(hyperparameters):
    """
	Generates a TensorFlow model with specified hyperparameters.

	:param hyperparameters: A dictionary containing the hyperparameters for the model.
	                        It should include the following keys:
	                        - 'dense_unit': The number of units in the dense layers.
	                        - 'num_hidden_layers': The number of hidden layers in the model.
	                        - 'dropout_rate': The dropout rate for the dense layers.
	                        - 'learning_rate': The learning rate for the Adam optimizer.
	:return: A compiled TensorFlow model.
	"""
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
    deep = layers.Dense(
        hyperparameters['dense_unit'],
        activation=tf.nn.relu
    )(concatenate)

    for _ in range(hyperparameters['num_hidden_layers']):
        deep = layers.Dense(
            hyperparameters['dense_unit'],
            activation=tf.nn.relu
        )(deep)
        deep = layers.Dropout(
            hyperparameters['dropout_rate']
        )(deep)

    outputs = layers.Dense(1, activation=tf.nn.sigmoid)(deep)

    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hyperparameters['learning_rate']
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    model.summary()

    return model


def run_fn(fn_args):
    """
	Runs the training process for a TensorFlow model.

	:param fn_args: An object containing the arguments for the function. 
    It should have the following attributes:
	                - hyperparameters: A dictionary containing the hyperparameters for the model.
	                - serving_model_dir: The directory where the serving model will be saved.
	                - transform_output: The output of the TensorFlow transform process.
	                - train_files: A list of training data files.
	                - eval_files: A list of evaluation data files.
	                - train_steps: The number of training steps.
	                - eval_steps: The number of evaluation steps.

	:return: None
	"""
    hyperparameters = fn_args.hyperparameters['values']
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_set = input_fn(fn_args.train_files, tf_transform_output)
    eval_set = input_fn(fn_args.eval_files, tf_transform_output)

    model = get_model(hyperparameters)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        update_freq='batch',
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy',
        mode='max',
        verbose=1,
        patience=10,
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor='val_binary_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True,
    )

    callbacks = [
        tensorboard_callback,
        early_stopping,
        model_checkpoint_callback
    ]

    model.fit(
        x=train_set,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_set,
        validation_steps=fn_args.eval_steps,
        callbacks=callbacks,
        verbose=1,
    )

    signatures = {
        'serving_default': get_serve_tf_examples_fn(
            model,
            tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'
            )
        )
    }

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures
    )
