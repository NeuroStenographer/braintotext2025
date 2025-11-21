"""
DaskRunner: A Kedro AbstractRunner implementation that distributes
nodes across a Dask cluster. Supports both local (Laptop) mode
via LocalCluster and remote scheduler mode via `address` in params.
"""

from collections import Counter
from itertools import chain
from typing import Any, Dict

from distributed import Client, LocalCluster, as_completed, worker_client

from kedro.framework.hooks.manager import (
    _create_hook_manager,
    _register_hooks,
    _register_hooks_entry_points,
)
from kedro.framework.project import settings
from kedro.io import AbstractDataset, DataCatalog
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from kedro.runner import AbstractRunner, run_node
from pluggy import PluginManager


class _DaskDataset(AbstractDataset):
    """Publish or retrieve datasets stored on the Dask scheduler."""

    def __init__(self, name: str):
        self._name = name

    def _load(self) -> Any:
        try:
            with worker_client() as client:
                return client.get_dataset(self._name)
        except ValueError:
            # If the scheduler holds the data
            return Client.current().get_dataset(self._name)

    def _save(self, data: Any) -> None:
        with worker_client() as client:
            client.publish_dataset(data, name=self._name, override=True)

    def _exists(self) -> bool:
        return self._name in Client.current().list_datasets()

    def _release(self) -> None:
        Client.current().unpublish_dataset(self._name)

    def _describe(self) -> Dict[str, Any]:
        return dict(name=self._name)


class DaskRunner(AbstractRunner):
    """
    Kedro runner that distributes node execution across a Dask cluster.

    Supports:
    - Remote Dask scheduler via `dask_client: { address: ... }`
    - Local Dask cluster via LocalCluster if no address is provided
    """

    def __init__(self, client_args: Dict[str, Any] = None, is_async: bool = False):
        super().__init__(is_async=is_async)
        client_args = client_args or {}

        # Case 1: Connect to remote scheduler
        if "address" in client_args and client_args["address"]:
            self._client = Client(**client_args)

        # Case 2: No address → run locally with LocalCluster
        else:
            cluster = LocalCluster(**client_args)
            self._client = Client(cluster)

        print(f"[DaskRunner] Connected | Dashboard: {self._client.dashboard_link}")

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass

    def create_default_dataset(self, ds_name: str) -> _DaskDataset:
        """Provide the default dataset type for missing datasets."""
        return _DaskDataset(ds_name)

    @staticmethod
    def _run_node(
        node: Node,
        catalog: DataCatalog,
        is_async: bool = False,
        session_id: str = None,
        *dependencies: Node,
    ) -> Node:
        """
        Execute a Kedro Node inside a Dask worker.

        Must re-create the HookManager on each worker (non-serializable).
        """
        hook_manager = _create_hook_manager()
        _register_hooks(hook_manager, settings.HOOKS)
        _register_hooks_entry_points(hook_manager, settings.DISABLE_HOOKS_FOR_PLUGINS)

        return run_node(node, catalog, hook_manager, is_async, session_id)

    def _run(
        self,
        pipeline: Pipeline,
        catalog: DataCatalog,
        hook_manager: PluginManager,
        session_id: str = None,
    ) -> None:
        """
        Orchestrates distributed execution of pipeline nodes.
        Handles dependency scheduling + dataset cleanup.
        """

        nodes = pipeline.nodes
        load_counts = Counter(chain.from_iterable(n.inputs for n in nodes))
        node_dependencies = pipeline.node_dependencies
        futures = {}

        client = self._client

        # Schedule nodes on the cluster
        for node in nodes:
            deps = (futures[d] for d in node_dependencies[node])
            futures[node] = client.submit(
                DaskRunner._run_node,
                node,
                catalog,
                self._is_async,
                session_id,
                *deps,
            )

        # Wait for completion
        for i, (_, node) in enumerate(as_completed(futures.values(), with_results=True)):
            self._logger.info("Completed node: %s", node.name)
            self._logger.info("Completed %d of %d tasks", i + 1, len(nodes))

            # Cleanup datasets no longer needed
            for dataset in node.inputs:
                load_counts[dataset] -= 1
                if load_counts[dataset] < 1 and dataset not in pipeline.inputs():
                    catalog.release(dataset)

            for dataset in node.outputs:
                if load_counts[dataset] < 1 and dataset not in pipeline.outputs():
                    catalog.release(dataset)

    def run_only_missing(self, pipeline: Pipeline, catalog: DataCatalog) -> Dict[str, Any]:
        """
        Run only missing or incomplete parts of the pipeline.
        Matches behavior of SequentialRunner.
        """
        free_outputs = pipeline.outputs() - set(catalog.list())
        missing = {ds for ds in catalog.list() if not catalog.exists(ds)}
        to_build = free_outputs | missing

        to_rerun = pipeline.only_nodes_with_outputs(*to_build) + pipeline.from_inputs(
            *to_build
        )

        unregistered_ds = pipeline.datasets() - set(catalog.list())
        missing_unregistered_ds = {
            ds for ds in unregistered_ds if not self.create_default_dataset(ds).exists()
        }

        output_to_unregistered = pipeline.only_nodes_with_outputs(
            *missing_unregistered_ds
        )
        needed_inputs = to_rerun.inputs() & missing_unregistered_ds
        to_rerun += output_to_unregistered.to_outputs(*needed_inputs)

        catalog = catalog.shallow_copy()
        for ds_name in unregistered_ds - missing_unregistered_ds:
            catalog.add(ds_name, self.create_default_dataset(ds_name))

        return self.run(to_rerun, catalog)
