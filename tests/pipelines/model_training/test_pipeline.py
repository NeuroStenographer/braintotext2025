from kedro.pipeline import Pipeline

from braintotext2025.pipelines.model_training.pipeline import create_pipeline


def test_model_training_pipeline_is_pipeline_instance():
    """Pipeline factory should return a Kedro Pipeline."""
    pipeline = create_pipeline()
    assert isinstance(pipeline, Pipeline)


def test_model_training_pipeline_has_expected_nodes():
    """Model training pipeline should have exactly one main node."""
    pipeline = create_pipeline()
    assert len(pipeline.nodes) == 1

    node = next(iter(pipeline.nodes))
    assert node.name == "train_validate_test_and_submit"


def test_model_training_pipeline_inputs_and_outputs():
    """
    Check the high-level inputs and outputs of the pipeline.
    This verifies wiring without being too fragile about internals.
    """
    pipeline = create_pipeline()

    # pipeline.inputs() / outputs() return sets of dataset names / params
    inputs = pipeline.inputs()
    outputs = pipeline.outputs()

    # These should match what we wired in model_training/pipeline.py:
    #   inputs=dict(
    #       train_loader="train_loader",
    #       val_loader="val_loader",
    #       test_loader="test_loader",
    #       params_paths="params:paths",
    #       params_train="params:train",
    #   ),
    #   outputs=["metrics", "results_table", "submission_csv"]
    expected_inputs = {
        "train_loader",
        "val_loader",
        "test_loader",
        "params:paths",
        "params:train",
    }
    expected_outputs = {
        "metrics",
        "results_table",
        "submission_csv",
    }

    # allow for exact match (preferred)
    assert inputs == expected_inputs
    assert outputs == expected_outputs
