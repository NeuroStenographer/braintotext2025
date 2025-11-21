"""Project hooks for braintotext2025."""

from typing import Any, Dict, Iterable, Optional

from kedro.config import ConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.versioning import Journal


class ProjectHooks:
    """Minimal project hooks.

    No Spark, no Dask-specific magic here – just the standard
    config loader and catalog wiring. Dask is handled via the
    runner and parameters, not via hooks.
    """

    @hook_impl
    def register_config_loader(
        self,
        conf_paths: Iterable[str],
        env: str,
        extra_params: Dict[str, Any],
    ) -> ConfigLoader:
        # Use Kedro's standard ConfigLoader; you still get env + extra_params.
        return ConfigLoader(conf_paths, env=env, extra_params=extra_params)

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
        journal: Journal,
    ) -> DataCatalog:
        # Standard catalog construction
        return DataCatalog.from_config(
            catalog,
            credentials,
            load_versions,
            save_version,
            journal,
        )
