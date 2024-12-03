import sys
import os

from rich.console import Console

from src.app import App

console = Console()

if __name__ == '__main__':
    try:
        app = App(sys.argv[1:])
        app.run()

    except Exception as e:
        console.print_exception()
        sys.exit(1)
