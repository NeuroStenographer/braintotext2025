# src/braintotext2025/hooks.py
"""Project hooks for braintotext2025.

We don't customise the config loader or catalog here; the default
OmegaConfigLoader and DataCatalog from Kedro are used.

Dask integration is handled via the custom DaskRunner and CLI,
not via hooks.
"""


class ProjectHooks:
    """Empty hook container referenced from settings.HOOKS.

    You can add hook_impl methods here later if you need them
    (before/after_node_run, before_pipeline_run, etc.), but for
    now this is intentionally minimal.
    """

    pass
