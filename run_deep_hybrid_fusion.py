"""Evaluate EEG and behavioral-feature attention."""

from deception.cli import legacy_main


if __name__ == "__main__":
    legacy_main('neural', '--method', 'attention', '--secondary', 'behavior')
