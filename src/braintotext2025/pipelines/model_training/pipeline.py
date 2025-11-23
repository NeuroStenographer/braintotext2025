# src/braintotext2025/pipelines/model_training/pipeline.py

from __future__ import annotations

from kedro.pipeline import node, pipeline
from .nodes import run_experiments


def create_pipeline(**_):
    """
    Pipeline for model training + validation + test +
    submission generation.
    """
    return pipeline([
        node(
            func=run_experiments,
            inputs={
                "train_loader": "train_loader",
                "val_loader": "val_loader",
                "test_loader": "test_loader",
                "params_paths": "params:paths",
                "params_train": "params:train",
            },
            outputs=[
                "metrics",
                "results_table",
                "submission_csv",
            ],
            name="train_validate_test_and_submit",
        ),
    ])
