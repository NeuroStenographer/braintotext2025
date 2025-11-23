from __future__ import annotations

from kedro.pipeline import Pipeline
from braintotext2025.pipelines import data_processing, model_training


def register_pipelines() -> dict[str, Pipeline]:
    dp = data_processing.create_pipeline()
    mt = model_training.create_pipeline()

    return {
        "__default__": dp + mt,
        "data_processing": dp,
        "model_training": mt,
    }
