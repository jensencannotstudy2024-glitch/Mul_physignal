"""Compare a teacher, a student, and a distilled student."""

from deception.cli import legacy_main


if __name__ == "__main__":
    legacy_main('neural', '--method', 'distillation', '--secondary', 'behavior')
