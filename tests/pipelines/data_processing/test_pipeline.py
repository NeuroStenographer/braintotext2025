from kedro.pipeline import Pipeline
from braintotext2025.pipelines.data_processing.pipeline import create_pipeline


def test_data_processing_pipeline_structure():
    p = create_pipeline()
    # basic sanity check
    assert isinstance(p, Pipeline)
    assert len(p.nodes) == 2

    # external inputs (params only)
    expected_inputs = {"params:paths.data_dir", "params:loader"}
    assert p.inputs() == expected_inputs

    # outputs from second node
    expected_outputs = {"train_loader", "val_loader", "test_loader"}
    assert p.outputs() == expected_outputs
