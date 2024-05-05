"""
Author: BilhaqAD
Date: 05/05/2024
Here's the transform module.
Usage:
- FOR TRANSFORM FEATURE OF DATASET
"""
import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURES = {
    "gender": 2,
    "smoking_history": 3
}

NUMERICAL_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]

LABEL_KEY = "diabetes"


def transformed_name(key):
    """
    Renames a feature key by appending '_xf' to it.

    Args:
        key (str): The original feature key.

    Returns:
        str: The transformed feature key with '_xf' appended to it.
    """

    return key + '_xf'


def convert_num_to_one_hot(label_tensor, num_labels=2):
    """
        Convert a label (0 or 1) into a one-hot vector
        Args:
            int: label_tensor (0 or 1)
        Returns
            label tensor
        """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def preprocessing_fn(inputs):
    """
    Preprocesses the input data by applying transformations to categorical and numerical features.

    Args:
        inputs (dict): A dictionary containing the input data. The keys are 
        the feature names and the values are the corresponding feature values.

    Returns:
        dict: A dictionary containing the preprocessed data. The keys are the transformed 
        feature names and the values are the transformed feature values.

    """
    outputs = {}

    for key, dim in CATEGORICAL_FEATURES.items():
        int_value = tft.compute_and_apply_vocabulary(
            inputs[key], top_k=dim + 1
        )
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )

    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
