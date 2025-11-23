"""Custom DaskRunner for braintotext2025.

This is adapted to Kedro 0.19.x, which uses `Task` for node execution
instead of the old `run_node` helper.
"""

from collections import Counter
from itertools import chain
from typing import Any, Dict

from distributed import Client, as_completed, worker_client

from kedro.framework.hooks.manager import (
    _create_hook_manager,
    _register_hooks,
    _register_hooks_entry_points,
)
from kedro.framework.project import settings
from kedro.io import AbstractDataset, DataCatalog
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from kedro.runner import AbstractRunner
from kedro.runner.task import Task
from pluggy import PluginManager


class _DaskDataset(AbstractDataset):
    """Dataset wrapper that publishes/gets named objects via the Dask scheduler."""

    def __init__(self, name: str):
        self._name = name

    def _load(self) -> Any:
        try:
            # When running inside a worker
            with worker_client() as client:
                return client.get_dataset(self._name)
        except ValueError:
            # When loading from the scheduler process
            return Client.current().get_dataset(self._name)

    def _save(self, data: Any) -> None:
        with worker_client() as client:
            client.publish_dataset(data, name=self._name, override=True)

    def _exists(self) -> bool:
        return self._name in Client.current().list_datasets()

    def _release(self) -> None:
        Client.current().unpublish_dataset(self._name)

    def _describe(self) -> Dict[str, Any]:
        return {"name": self._name}


class DaskRunner(AbstractRunner):
    """Runner that executes Kedro nodes on a Dask cluster."""

    def __init__(self, client_args: Dict[str, Any] | None = None, is_async: bool = False):
        """Create a Dask client and register the runner.

        Args:
            client_args: kwargs passed to ``distributed.Client(**client_args)``.
                         If None or empty, this creates a local Dask cluster.
            is_async: whether to load/save datasets asynchronously.
        """
        super().__init__(is_async=is_async)
        client_args = client_args or {}
        Client(**client_args)

    def __del__(self):
        # Ensure the client is cleaned up when the runner is GC'd
        try:
            Client.current().close()
        except Exception:
            pass

    # ---- AbstractRunner API requirements ---------------------------------

    def _get_executor(self, max_workers: int):
        """We don't use the base Executor-based scheduling; return None."""
        return None

    def create_default_dataset(self, ds_name: str) -> _DaskDataset:
        """Create the default dataset for unregistered intermediate outputs."""
        return _DaskDataset(ds_name)

    # ---- Internal helper used on Dask workers ----------------------------

    @staticmethod
    def _run_node(
        node: Node,
        catalog: DataCatalog,
        is_async: bool = False,
        run_id: str | None = None,
        *dependencies: Node,
    ) -> Node:
        """Execute a single node on a Dask worker.

        We recreate a hook manager on the worker, just like the old
        kedro-dask implementation did.
        """
        hook_manager = _create_hook_manager()
        _register_hooks(hook_manager, settings.HOOKS)
        _register_hooks_entry_points(hook_manager, settings.DISABLE_HOOKS_FOR_PLUGINS)

        task = Task(
            node=node,
            catalog=catalog,
            hook_manager=hook_manager,
            is_async=is_async,
            run_id=run_id,
        )
        # Dask handles process/threading, so we don't need Task.parallel here.
        return task.execute()

    # ---- Core Dask scheduling logic --------------------------------------

    def _run(
        self,
        pipeline: Pipeline,
        catalog: DataCatalog,
        hook_manager: PluginManager | None = None,
        run_id: str | None = None,
    ) -> None:
        """Schedule all nodes on the Dask cluster."""

        nodes = pipeline.nodes
        load_counts = Counter(chain.from_iterable(n.inputs for n in nodes))
        node_dependencies = pipeline.node_dependencies
        node_futures: dict[Node, Any] = {}

        client = Client.current()

        # Submit nodes with proper dependency futures
        for node in nodes:
            dependencies = (node_futures[dep] for dep in node_dependencies[node])
            node_futures[node] = client.submit(
                DaskRunner._run_node,
                node,
                catalog,
                self._is_async,
                run_id,
                *dependencies,
            )

        # Wait for completion and release datasets as we go
        for i, (future, node) in enumerate(
            as_completed(node_futures.values(), with_results=True)
        ):
            # `as_completed(..., with_results=True)` yields (future, result),
            # and our `_run_node` returns the Node object.
            self._logger.info("Completed node: %s", node.name)
            self._logger.info("Completed %d out of %d tasks", i + 1, len(nodes))

            # Release inputs that are no longer needed
            for dataset in node.inputs:
                load_counts[dataset] -= 1
                if load_counts[dataset] < 1 and dataset not in pipeline.inputs():
                    catalog.release(dataset)

            # Release outputs that are no longer needed
            for dataset in node.outputs:
                if load_counts[dataset] < 1 and dataset not in pipeline.outputs():
                    catalog.release(dataset)
