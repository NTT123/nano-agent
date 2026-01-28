"""Interactive UI elements for the CLI.

This module provides an abstraction for interactive terminal elements
that can control output and capture input.

Usage:
    from cli.elements import ElementManager, ConfirmPrompt, MultiLinePrompt

    manager = ElementManager()

    # Get user confirmation
    result = await manager.run(ConfirmPrompt(
        message="Apply changes?",
        preview_lines=["- old", "+ new"]
    ))

    # Get text input with multiline support (\\ + Enter)
    text = await manager.run(MultiLinePrompt(prompt="> "))

    # Using unified footer system
    from cli.elements import TerminalFooter, FooterElementManager, FooterInput

    footer = TerminalFooter()
    footer.activate()
    manager = FooterElementManager(footer)
    text = await manager.run(FooterInput(prompt="> "))
"""

from .base import ActiveElement, InputEvent
from .confirm_prompt import ConfirmPrompt
from .footer import StatusBarState, TerminalFooter
from .footer_input import FooterInput
from .footer_manager import FooterElementManager
from .manager import ElementManager
from .menu_select import MenuSelect
from .terminal import RawInputReader, TerminalRegion
from .prompt_toolkit_input import PromptToolkitInput
from .text_prompt import MultiLinePrompt, TextPrompt

__all__ = [
    # Base
    "ActiveElement",
    "InputEvent",
    # Managers
    "ElementManager",
    "FooterElementManager",
    # Footer
    "TerminalFooter",
    "StatusBarState",
    # Terminal
    "TerminalRegion",
    "RawInputReader",
    # Elements
    "TextPrompt",
    "MultiLinePrompt",
    "FooterInput",
    "PromptToolkitInput",
    "ConfirmPrompt",
    "MenuSelect",
]
