"""
    Shortcuts
"""
import sys

import pynput.keyboard
from pynput import keyboard

from snip2fumen.snipe2fumen import snipe_and_recog, recog_from_clipboard


def _snipe(to_clipboard: bool):
    """
        Callback for snipe to fumen
    """
    return lambda: snipe_and_recog(to_clipboard)


def _clipboard_to_fumen(to_clipboard: bool):
    """
        Callback for clipboard to fumen
    """
    return lambda: recog_from_clipboard(to_clipboard)


def _stop():
    sys.exit()


def listen_for_shortcuts(to_clipboard: bool = False):
    """
        Block the program and listen for shortcuts
    """
    print("listening...")
    with keyboard.GlobalHotKeys({
            '<ctrl>+<shift>+C': _snipe(to_clipboard),
            '<ctrl>+<shift>+D': _clipboard_to_fumen(to_clipboard),
            '<ctrl>+Q': _stop}) as hotkeys:
        hotkeys.join()
