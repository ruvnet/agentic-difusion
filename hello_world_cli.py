#!/usr/bin/env python3
"""
A more sophisticated Hello World CLI application with command-line arguments.
"""

import argparse
import sys


def create_parser():
    """Create the command line parser."""
    parser = argparse.ArgumentParser(
        description="A simple Hello World CLI program with customization options."
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="World",
        help="Name to greet (default: 'World')"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "es", "fr", "de", "zh"],
        default="en",
        help="Language for greeting (default: 'en')"
    )
    
    parser.add_argument(
        "--uppercase",
        action="store_true",
        help="Display the greeting in uppercase"
    )
    
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the greeting (default: 1)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
        help="Show program's version number and exit"
    )
    
    return parser


def get_greeting(language, name):
    """Return greeting in the specified language."""
    greetings = {
        "en": f"Hello, {name}!",
        "es": f"¡Hola, {name}!",
        "fr": f"Bonjour, {name}!",
        "de": f"Hallo, {name}!",
        "zh": f"你好, {name}!"
    }
    return greetings.get(language, greetings["en"])


def main():
    """Main entry point for the CLI application."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Generate greeting
    greeting = get_greeting(args.language, args.name)
    
    # Apply uppercase if requested
    if args.uppercase:
        greeting = greeting.upper()
    
    # Print greeting the specified number of times
    for _ in range(args.repeat):
        print(greeting)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())