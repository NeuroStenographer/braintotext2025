# src/braintotext2025/cli.py
from __future__ import annotations

import click
from kedro.framework.cli.utils import CONTEXT_SETTINGS
from kedro.framework.cli.project import run as kedro_run

@click.group(context_settings=CONTEXT_SETTINGS)
def cli() -> None:
    """Project CLI group for braintotext2025."""
    pass

# expose the standard Kedro run command
cli.add_command(kedro_run)

# so utils.find_run_command(...) still works
run = kedro_run
