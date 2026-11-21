# -*- coding: utf-8 -*-
"""
Top-level CLI for ecosound.

Provides the ``ecosound`` console command with subcommand groups for each
module (e.g. ``ecosound qaqc run ...``).

Usage
-----
.. code-block:: bash

   ecosound --help
   ecosound --version
   ecosound qaqc --help
"""

import click

from ecosound._version import __version__


@click.group()
@click.version_option(version=__version__, prog_name="ecosound")
def main():
    """ecosound -- Python toolkit for passive acoustic data analysis."""


# Register subcommand groups -------------------------------------------------
from ecosound.soundscape.cli_qaqc import qaqc  # noqa: E402

main.add_command(qaqc)
