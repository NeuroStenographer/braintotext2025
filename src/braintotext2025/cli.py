"""Project-specific Kedro CLI for braintotext2025."""

import click
from kedro.framework.cli.project import (
    ASYNC_ARG_HELP,
    CONFIG_FILE_HELP,
    CONF_SOURCE_HELP,
    FROM_INPUTS_HELP,
    FROM_NODES_HELP,
    LOAD_VERSION_HELP,
    NODE_ARG_HELP,
    PARAMS_ARG_HELP,
    PIPELINE_ARG_HELP,
    RUNNER_ARG_HELP,
    TAG_ARG_HELP,
    TO_NODES_HELP,
    TO_OUTPUTS_HELP,
    project_group,
)
from kedro.framework.cli.utils import (
    CONTEXT_SETTINGS,
    _config_file_callback,
    _get_values_as_tuple,
    _reformat_load_versions,
    _split_params,
    env_option,
    split_string,
    split_node_names,
)
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.utils import load_obj


@click.group(context_settings=CONTEXT_SETTINGS, name=__package__)
def cli() -> None:
    """Project-specific Kedro CLI for braintotext2025."""


@project_group.command()
@env_option
@click.option(
    "--node",
    "-n",
    "node_names",
    type=str,
    multiple=True,
    help=NODE_ARG_HELP,
)
@click.option(
    "--runner",
    "-r",
    type=str,
    default=None,
    help=RUNNER_ARG_HELP,
)
@click.option(
    "--async",
    "is_async",
    is_flag=True,
    help=ASYNC_ARG_HELP,
)
@click.option(
    "--tag",
    "-t",
    type=str,
    multiple=True,
    help=TAG_ARG_HELP,
)
@click.option(
    "--from-inputs",
    type=str,
    default="",
    help=FROM_INPUTS_HELP,
    callback=split_string,
)
@click.option(
    "--to-outputs",
    type=str,
    default="",
    help=TO_OUTPUTS_HELP,
    callback=split_string,
)
@click.option(
    "--from-nodes",
    type=str,
    default="",
    help=FROM_NODES_HELP,
    callback=split_node_names,
)
@click.option(
    "--to-nodes",
    type=str,
    default="",
    help=TO_NODES_HELP,
    callback=split_node_names,
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    help=CONFIG_FILE_HELP,
    callback=_config_file_callback,
)
@click.option(
    "--load-version",
    "-lv",
    type=str,
    multiple=True,
    help=LOAD_VERSION_HELP,
)
@click.option(
    "--pipeline",
    "-p",
    type=str,
    default=None,
    help=PIPELINE_ARG_HELP,
)
@click.option(
    "--params",
    type=str,
    multiple=True,
    help=PARAMS_ARG_HELP,
)
@click.option(
    "--conf-source",
    type=str,
    default=None,
    help=CONF_SOURCE_HELP,
)
def run(  # noqa: PLR0913 (many args – matches Kedro template)
    env,
    node_names,
    runner,
    is_async,
    tag,
    from_inputs,
    to_outputs,
    from_nodes,
    to_nodes,
    config,
    load_version,
    pipeline,
    params,
    conf_source,
):
    """Run the pipeline (supports DaskRunner)."""

    # Make sure project settings are bootstrapped
    bootstrap_project("braintotext2025")

    # default runner if none passed
    runner = runner or "SequentialRunner"

    # normalise CLI inputs like Kedro’s own run command
    tag = _get_values_as_tuple(tag) if tag else tag
    node_names = _get_values_as_tuple(node_names) if node_names else node_names
    load_version = _reformat_load_versions(load_version)
    params = _split_params(params)

    with KedroSession.create(
        env=env,
        extra_params=params,
        conf_source=conf_source,
    ) as session:
        context = session.load_context()
        runner_instance = _instantiate_runner(runner, is_async, context)
        session.run(
            tags=tag,
            runner=runner_instance,
            node_names=node_names,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            from_inputs=from_inputs,
            to_outputs=to_outputs,
            load_versions=load_version,
            pipeline_name=pipeline,
        )


def _instantiate_runner(runner: str, is_async: bool, project_context):
    """Instantiate runner and inject Dask client args if using DaskRunner."""
    runner_class = load_obj(runner, "kedro.runner")
    runner_kwargs = dict(is_async=is_async)

    # This is the bit from the docs: read dask_client from your params
    if runner.endswith("DaskRunner"):
        client_args = project_context.params.get("dask_client") or {}
        runner_kwargs.update(client_args=client_args)

    return runner_class(**runner_kwargs)
