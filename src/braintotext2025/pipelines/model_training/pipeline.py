from kedro.pipeline import node, pipeline
from .nodes import run_experiments

def create_pipeline(**_):
    return pipeline([
        node(
            func=run_experiments,
            inputs=dict(
                train_loader="train_loader",
                val_loader="val_loader",
                test_loader="test_loader",
                params_paths="params:paths",
                params_train="params:train",
            ),
            outputs=["metrics","results_table","submission_csv"],
            name="train_validate_test_and_submit",
        ),
    ])
