"""Evaluate direct LLM prompts with explicit failures and coverage."""

from deception.cli import legacy_main


if __name__ == "__main__":
    legacy_main('llm', '--mode', 'direct')
