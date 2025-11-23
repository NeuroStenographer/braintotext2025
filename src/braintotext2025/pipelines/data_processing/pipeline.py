from kedro.pipeline import node, pipeline
from .nodes import build_datasets, build_dataloaders


def create_pipeline(**_):
    return pipeline(
        [
            node(
                func=build_datasets,
                inputs="params:paths.data_dir",
                outputs=["train_dataset", "val_dataset", "test_dataset"],
                name="build_datasets_node",
                tags=["data", "datasets"],
            ),
            node(
                func=build_dataloaders,
                inputs=[
                    "train_dataset",
                    "val_dataset",
                    "test_dataset",
                    "params:loader",
                ],
                outputs=["train_loader", "val_loader", "test_loader"],
                name="build_dataloaders_node",
                tags=["data", "dataloaders"],
            ),
        ]
    )
