"""
Author: BilhaqAD
Date: 05/05/2024
Here's the pipline module.
Usage:
- FOR INIT & RUN PIPELINE
"""
import os
from typing import Text

from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from components import init_components

PIPELINE_NAME = "crossnexx-pipeline"

DATA_ROOT = "data"
TRAMSFORM_MODULE_FILE = "modules/transform.py"
TUNER_MODULE_FILE = "module/tuner.py"
TRAINER_MODULE_FILE = "module/trainer.py"

OUTPUT_BASE = "outputs"
serving_model_dir = os.path.join(OUTPUT_BASE, "serving_model")
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")


def init_local_pipeline(
        components_param,
        pipeline_root_param: Text
) -> pipeline.Pipeline:  # type: ignore
    """
    Initializes a TFX pipeline with the given components and pipeline root.

    Args:
        components (List[base_component.BaseComponent]): A list of TFX components.
        pipeline_root_param (str): The root directory of the pipeline.

    Returns:
        pipeline.Pipeline: The initialized TFX pipeline.

    Raises:
        None

    Examples:
        >>> components_param = [CsvExampleGen(), SchemaGen(), Transform()]
        >>> pipeline_root_param = "./pipeline"
        >>> init_local_pipeline(components, pipeline_root)
        Pipeline(pipeline_name='crossnexx-pipeline', 
        pipeline_root_param='./pipeline', 
        components_param=[CsvExampleGen, 
        SchemaGen, Transform], 
        enable_cache=True, 
        metadata_connection_config=MetadataConnectionConfig(
            metadata_path='./pipeline/metadata.sqlite'
        ), 
        beam_pipeline_args=['--direct_running_mode=multi_processing', 
        '----direct_num_workers=0'])
    """

    logging.info(f"Pipeline root set to: {pipeline_root_param}")
    beam_args = [
        "--direct_running_mode=multi_processing",
        # 0 auto-detect based on the number of CPUs available
        "----direct_num_workers=0"
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root_param,
        components=components_param,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_args
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    components = init_components(
        DATA_ROOT,
        transform_module=TRAMSFORM_MODULE_FILE,
        tuner_module=TUNER_MODULE_FILE,
        training_module=TRAINER_MODULE_FILE,
        train_steps=5000,
        eval_steps=1000,
        serving_model_dir=serving_model_dir
    )

    pipeline = init_local_pipeline(components, pipeline_root)
    BeamDagRunner().run(pipeline)
