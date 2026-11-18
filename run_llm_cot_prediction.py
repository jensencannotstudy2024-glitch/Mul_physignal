"""Evaluate training-participant few-shot examples without invented physiology."""

from deception.cli import legacy_main


if __name__ == "__main__":
    legacy_main('llm', '--mode', 'few_shot')
