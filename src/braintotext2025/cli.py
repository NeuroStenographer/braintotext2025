from __future__ import annotations

from typing import Any, Dict, Optional

import click
from kedro.framework.cli.project import project_group
from kedro.framework.cli.utils import _get_values_as_tuple
from kedro.framework.session import KedroSession
from kedro.utils import load_obj


# Re-export the project_group as "cli" so Kedro can find it
cli = project_group


@cli.command(name="run")
@click.option("--env", "-e", type=str, default=None, help="Kedro environment.")
@click.option(
    "--runner",
    "-r",
    type=str,
    default=None,
    help=(
        "Runner class to use, e.g. 'SequentialRunner', 'ParallelRunner', "
        "or 'braintotext2025.runner.DaskRunner'."
    ),
)
@click.option(
    "--is-async",
    is_flag=True,
    default=False,
    help="Whether to use the runner in asynchronous mode (if supported).",
)
@click.option(
    "--pipeline",
    "-p",
    "pipeline_name",
    type=str,
    default=None,
    help="Name of the pipeline to run (defaults to '__default__').",
)
@click.option(
    "--tag",
    "-t",
    "tags",
    multiple=True,
    help="Run only nodes with these tag(s). Can be used multiple times.",
)
@click.option(
    "--node",
    "-n",
    "node_names",
    multiple=True,
    help="Run only node(s) with these name(s). Can be used multiple times.",
)
@click.option(
    "--from-nodes",
    multiple=True,
    help="Run from these node(s) onward. Can be used multiple times.",
)
@click.option(
    "--to-nodes",
    multiple=True,
    help="Run up to these node(s). Can be used multiple times.",
)
@click.option(
    "--from-inputs",
    multiple=True,
    help="Run only nodes dependent on these dataset(s).",
)
@click.option(
    "--to-outputs",
    multiple=True,
    help="Run only nodes which are ancestors of these dataset(s).",
)
@click.option(
    "--load-version",
    type=str,
    default=None,
    help="Specify a dataset version to load.",
)
@click.option(
    "--params",
    type=str,
    multiple=True,
    help=(
        "Extra parameters to override config values, "
        "e.g. --params train.epochs=1 --params train.lr=0.0001"
    ),
)
def run(
    env: Optional[str],
    runner: Optional[str],
    is_async: bool,
    pipeline_name: Optional[str],
    tags,
    node_names,
    from_nodes,
    to_nodes,
    from_inputs,
    to_outputs,
    load_version: Optional[str],
    params,
) -> None:
    """Run the Kedro pipeline, optionally using a Dask runner."""

    # Default: if runner is not specified, use the standard SequentialRunner.
    # To run with Dask, call:
    #   kedro run --runner=braintotext2025.runner.DaskRunner
    runner = runner or "SequentialRunner"

    # Convert click's multiple=True options into tuples
    tags = _get_values_as_tuple(tags) if tags else tags
    node_names = _get_values_as_tuple(node_names) if node_names else node_names

    extra_params: Dict[str, Any] = {}
    for p in params:
        if "=" in p:
            key, value = p.split("=", 1)
            extra_params[key] = value

    with KedroSession.create(env=env, extra_params=extra_params) as session:
        context = session.load_context()
        runner_instance = _instantiate_runner(runner, is_async, context)

        session.run(
            tags=tags,
            runner=runner_instance,
            node_names=node_names,
            from_nodes=from_nodes or None,
            to_nodes=to_nodes or None,
            from_inputs=from_inputs or None,
            to_outputs=to_outputs or None,
            load_versions=load_version,
            pipeline_name=pipeline_name,
        )


def _load_runner_class(runner: str):
    """Load a runner class.

    If `runner` contains a dot, treat it as a fully qualified path, e.g.:
        'braintotext2025.runner.DaskRunner'

    Otherwise, load it from the built-in 'kedro.runner' module, e.g.:
        'SequentialRunner', 'ParallelRunner'
    """
    if "." in runner:
        module_path, class_name = runner.rsplit(".", 1)
        return load_obj(class_name, module_path)
    else:
        return load_obj(runner, "kedro.runner")


def _instantiate_runner(runner: str, is_async: bool, project_context):
    """Instantiate the chosen runner, passing Dask config when relevant."""
    runner_class = _load_runner_class(runner)
    kwargs: Dict[str, Any] = {"is_async": is_async}

    # If you're using a custom DaskRunner (e.g. in braintotext2025.runner),
    # you can configure it from params.dask_client in your parameters.yml:
    #
    # dask_client:
    #   address: "tcp://127.0.0.1:8786"
    #   # ... other kwargs your DaskRunner expects
    #
    if "DaskRunner" in runner_class.__name__:
        dask_client_params = project_context.params.get("dask_client") or {}
        # Many DaskRunner implementations expect something like client_args=
        kwargs["client_args"] = dask_client_params

    return runner_class(**kwargs)
