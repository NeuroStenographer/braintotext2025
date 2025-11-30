# src/braintotext2025/hooks.py
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

from kedro.config import OmegaConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog

logger = logging.getLogger(__name__)

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
    """Project hooks."""

    @hook_impl
    def register_config_loader(
        self,
        conf_paths: Iterable[str],
        env: str,
        extra_params: Dict[str, Any],
    ) -> OmegaConfigLoader:
        return OmegaConfigLoader(
            conf_paths,
            env=env,
            base_env="base",
            extra_params=extra_params,
        )

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: Optional[str],
        **kwargs: Any,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog=catalog,
            credentials=credentials,
            load_versions=load_versions,
            save_version=save_version,
        )

    @hook_impl
    def before_pipeline_run(
        self,
        run_params: Dict[str, Any],
        pipeline: Any,
        catalog: DataCatalog,
    ) -> None:
        """Optionally initialise Ray and Dask-on-Ray before the pipeline run."""
        ray_cfg = run_params.get("ray") or {}
        logger.info("[HOOK] ray config from params: %r", ray_cfg)

        if not ray_cfg.get("enabled", False):
            logger.info("[HOOK] Ray disabled or not configured; skipping Ray init.")
            return

        if not _HAS_RAY:
            raise RuntimeError(
                "Ray is enabled in parameters, but `ray` or `ray[default]` "
                "is not installed in this environment."
            )

        init_kwargs = {
            k: v for k, v in ray_cfg.items()
            if k not in {"enabled"}
        }

        logger.info("[HOOK] Initializing Ray with kwargs: %r", init_kwargs)

        if not ray.is_initialized():
            ray.init(**init_kwargs)

        logger.info("[HOOK] Ray initialized: %s", ray.is_initialized())

        if enable_dask_on_ray is not None:
            enable_dask_on_ray()
            logger.info("[HOOK] Dask-on-Ray integration enabled.")
        else:
            logger.warning(
                "[HOOK] ray.util.dask.enable_dask_on_ray not available; "
                "Dask-on-Ray is NOT enabled."
            )
