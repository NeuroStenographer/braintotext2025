"""Project hooks for braintotext2025."""

from typing import Any, Dict, Iterable, Optional

from kedro.config import OmegaConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.versioning import Journal

# Optional Ray / Dask-on-Ray integration
try:
    import ray
    from ray.util.dask import enable_dask_on_ray
except ImportError:  # Ray not installed; we handle this gracefully below
    ray = None
    enable_dask_on_ray = None


class ProjectHooks:
    """Project hooks.

    - register_config_loader: use OmegaConfigLoader with env + extra_params
    - after_context_created: optionally enable Dask-on-Ray
    - register_catalog: construct the DataCatalog from config
    """

    @hook_impl
    def register_config_loader(
        self,
        conf_paths: Iterable[str],
        env: str,
        extra_params: Dict[str, Any],
    ) -> OmegaConfigLoader:
        return OmegaConfigLoader(conf_paths, env=env, extra_params=extra_params)

    @hook_impl
    def after_context_created(self, context) -> None:
        """Optionally enable Ray + Dask-on-Ray based on params:dask_client."""
        params = getattr(context, "params", {}) or {}
        dask_cfg: Dict[str, Any] = params.get("dask_client", {})

        use_ray = dask_cfg.get("use_ray", False)
        if not use_ray:
            return  # plain Dask, nothing special to do

        if ray is None or enable_dask_on_ray is None:
            raise ImportError(
                "You set params:dask_client.use_ray = true but Ray or "
                "ray.util.dask is not installed.\n"
                "Install with e.g. `pip install 'ray[default]' ray[dask]` and retry."
            )

        # Optional Ray init kwargs, e.g. num_cpus, address, etc.
        ray_init_kwargs: Dict[str, Any] = dask_cfg.get("ray_init", {})

        if not ray.is_initialized():
            ray.init(**ray_init_kwargs)

        enable_dask_on_ray()

        # Log a friendly message if context has a logger
        logger = getattr(context, "logger", None) or getattr(context, "_logger", None)
        if logger:
            logger.info("Dask-on-Ray backend enabled via ProjectHooks.")

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
        journal: Journal,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog,
            credentials,
            load_versions,
            save_version,
            journal,
        )
