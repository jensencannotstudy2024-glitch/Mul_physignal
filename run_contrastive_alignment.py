"""Compare alignment with an otherwise identical neural baseline."""

from deception.cli import legacy_main


if __name__ == "__main__":
    legacy_main('neural', '--method', 'contrastive', '--secondary', 'behavior')
