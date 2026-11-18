"""Evaluate late fusion with distinct named modalities."""

from deception.cli import legacy_main


if __name__ == "__main__":
    legacy_main('fusion', '--method', 'late')
