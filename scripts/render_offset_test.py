from __future__ import annotations

import sys
import time

from rich.console import Console
from rich.text import Text


def main() -> None:
    console = Console(soft_wrap=True)

    # Simulate a prompt without newline, then user input
    sys.stdout.write("> ")
    sys.stdout.flush()
    time.sleep(0.2)
    sys.stdout.write("hey")
    sys.stdout.flush()
    time.sleep(0.2)
    sys.stdout.write("\n")
    sys.stdout.flush()

    # Baseline: normal print without reset
    console.print("Baseline without reset:")
    console.print("Hello from Rich")
    console.print("Tokens: in=1 out=2 total=3")

    # With carriage return only
    console.print("\nWith carriage return:")
    console.print("\r", end="")
    console.print("Hello from Rich")
    console.print("Tokens: in=1 out=2 total=3")

    # With clear line + carriage return
    console.print("\nWith clear line + carriage return:")
    console.print("\r\033[2K", end="")
    console.print("Hello from Rich")
    console.print("Tokens: in=1 out=2 total=3")

    # Multi-line test
    console.print("\nMultiline test:")
    console.print("Line 1\nLine 2\nLine 3")
    console.print("Tokens: in=1 out=2 total=3")

    # Text object test
    console.print("\nText object test:")
    text = Text()
    text.append("Thinking: ", style="magenta")
    text.append("short thought", style="dim")
    console.print(text)
    console.print("-----", style="dim")
    console.print("Hello from Rich")
    console.print("Tokens: in=1 out=2 total=3")


if __name__ == "__main__":
    main()
