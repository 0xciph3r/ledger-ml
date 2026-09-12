#!/usr/bin/env python3
"""Container and local entrypoint for Ledger ML synthetic fraud training."""

from ledgerml_training.trainer import run_cli


if __name__ == "__main__":
    raise SystemExit(run_cli())
