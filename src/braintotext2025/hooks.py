# src/braintotext2025/hooks.py
"""Project hooks for braintotext2025."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from kedro.config import OmegaConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog

# Optional Ray / Dask-on-Ray integration
try:
    import ray  # type: ignore
    from ray.util.dask import enable_dask_on_ray  # type: ignore

    _HAS_RAY = True
except Exception:  # pragma: no cover
    ray = None  # type: ignore
    enable_dask_on_ray = None  # type: ignore
    _HAS_RAY = False


class ProjectHooks:
    """Project hooks.

    - Uses OmegaConfigLoader for env + extra params.
    - Builds a standard DataCatalog from YAML.
    - Optionally wires Dask-on-Ray if enabled in parameters.
    """

    # -----------------------------
    # Config loader
    # -----------------------------
    @hook_impl
    def register_config_loader(
        self,
        conf_paths: Iterable[str],
        env: str,
        extra_params: Dict[str, Any],
    ) -> OmegaConfigLoader:
        """Return the OmegaConfigLoader used by the project."""
        # This mirrors the default 0.19 project template
        return OmegaConfigLoader(
            conf_paths,
            env=env,
            base_env="base",
            extra_params=extra_params,
        )

    # -----------------------------
    # Catalog
    # -----------------------------
    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: Optional[str],
        **kwargs: Any,
    ) -> DataCatalog:
        """Build the project DataCatalog.

        Note: Journal was removed in Kedro 0.18+, so we do NOT import or accept it.
        **kwargs is here to stay forwards-compatible with any extra hook args.
        """
        return DataCatalog.from_config(
            catalog=catalog,
            credentials=credentials,
            load_versions=load_versions,
            save_version=save_version,
        )

    # -----------------------------
    # Optional: Ray + Dask-on-Ray
    # -----------------------------
    @hook_impl
    def before_pipeline_run(
        self,
        run_params: Dict[str, Any],
        pipeline: Any,
        catalog: DataCatalog,
    ) -> None:
        """Optionally initialise Ray and Dask-on-Ray before the pipeline run.

        Controlled via `params:ray` (all optional), e.g.:

        ray:
          enabled: true
          address: "auto"
          num_cpus: 8
          num_gpus: 0
        """
        ray_cfg = run_params.get("ray") or {}
        if not ray_cfg.get("enabled", False):
            return

        if not _HAS_RAY:
            raise RuntimeError(
                "Ray is enabled in parameters, but `ray` or `ray[default]` "
                "is not installed in this environment."
            )

        # Start / connect to Ray
        address = ray_cfg.get("address", None)
        init_kwargs = {k: v for k, v in ray_cfg.items() if k not in {"enabled"}}
        if address is not None:
            init_kwargs["address"] = address

        if not ray.is_initialized():
            ray.init(**init_kwargs)

        # Enable Dask-on-Ray if available
        if enable_dask_on_ray is not None:
            enable_dask_on_ray()
