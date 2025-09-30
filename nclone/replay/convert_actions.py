#!/usr/bin/env python3
"""
N++ Action Format Converter

Utility for converting between different N++ action representations.
This is useful when working with different replay formats or action encodings.

Usage:
    python -m nclone.replay.convert_actions --input actions.txt --output converted.txt --format symbols
"""

import argparse
from pathlib import Path
from typing import List


class ActionConverter:
    """Converts between different N++ action format representations."""

    # Action mapping from text names to symbols
    TEXT_TO_SYMBOL = {
        "NOOP": "-",  # Nothing
        "Jump": "^",  # Jump
        "Right": ">",  # Right
        "Jump + Right": "/",  # Right Jump
        "Left": "<",  # Left
        "Jump + Left": "\\",  # Left Jump
    }

    # Reverse mapping from symbols to text
    SYMBOL_TO_TEXT = {v: k for k, v in TEXT_TO_SYMBOL.items()}

    # Action mapping to discrete indices (matching NppEnvironment action space)
    TEXT_TO_INDEX = {
        "NOOP": 0,
        "Left": 1,
        "Right": 2,
        "Jump": 3,
        "Jump + Left": 4,
        "Jump + Right": 5,
    }

    # Reverse mapping from indices to text
    INDEX_TO_TEXT = {v: k for k, v in TEXT_TO_INDEX.items()}

    # Binary input mapping from attract file format (0-7) to text actions
    # Based on the N++ attract file demo data format
    BINARY_TO_TEXT = {
        0: "NOOP",  # No input
        1: "Jump",  # Jump only
        2: "Right",  # Right only
        3: "Jump + Right",  # Right + Jump
        4: "Left",  # Left only
        5: "Jump + Left",  # Left + Jump
        6: "Left",  # Left only (duplicate mapping)
        7: "Jump + Left",  # Left + Jump (duplicate mapping)
    }

    # Reverse mapping from text to binary format
    TEXT_TO_BINARY = {
        "NOOP": 0,
        "Jump": 1,
        "Right": 2,
        "Jump + Right": 3,
        "Left": 4,
        "Jump + Left": 5,
    }

    @classmethod
    def convert_text_to_symbol(cls, action: str) -> str:
        """Convert from text format to symbol format."""
        return cls.TEXT_TO_SYMBOL.get(action.strip(), action)

    @classmethod
    def convert_symbol_to_text(cls, symbol: str) -> str:
        """Convert from symbol format to text format."""
        return cls.SYMBOL_TO_TEXT.get(symbol.strip(), symbol)

    @classmethod
    def convert_text_to_index(cls, action: str) -> int:
        """Convert from text format to discrete index."""
        return cls.TEXT_TO_INDEX.get(action.strip(), 0)  # Default to NOOP

    @classmethod
    def convert_index_to_text(cls, index: int) -> str:
        """Convert from discrete index to text format."""
        return cls.INDEX_TO_TEXT.get(index, "NOOP")

    @classmethod
    def convert_symbol_to_index(cls, symbol: str) -> int:
        """Convert from symbol format to discrete index."""
        text = cls.convert_symbol_to_text(symbol)
        return cls.convert_text_to_index(text)

    @classmethod
    def convert_index_to_symbol(cls, index: int) -> str:
        """Convert from discrete index to symbol format."""
        text = cls.convert_index_to_text(index)
        return cls.convert_text_to_symbol(text)

    @classmethod
    def convert_binary_to_text(cls, binary_value: int) -> str:
        """Convert from binary format (0-7) to text format."""
        return cls.BINARY_TO_TEXT.get(binary_value, "NOOP")

    @classmethod
    def convert_text_to_binary(cls, action: str) -> int:
        """Convert from text format to binary format (0-7)."""
        return cls.TEXT_TO_BINARY.get(action.strip(), 0)

    @classmethod
    def convert_binary_to_symbol(cls, binary_value: int) -> str:
        """Convert from binary format (0-7) to symbol format."""
        text = cls.convert_binary_to_text(binary_value)
        return cls.convert_text_to_symbol(text)

    @classmethod
    def convert_binary_to_index(cls, binary_value: int) -> int:
        """Convert from binary format (0-7) to discrete index."""
        text = cls.convert_binary_to_text(binary_value)
        return cls.convert_text_to_index(text)


def convert_action_sequence(
    actions: List[str], input_format: str, output_format: str
) -> List[str]:
    """
    Convert a sequence of actions between formats.

    Args:
        actions: List of actions in input format
        input_format: Input format ('text', 'symbol', 'index', 'binary')
        output_format: Output format ('text', 'symbol', 'index', 'binary')

    Returns:
        List of actions in output format
    """
    converter = ActionConverter()
    converted = []

    for action in actions:
        if input_format == "text" and output_format == "symbol":
            converted.append(converter.convert_text_to_symbol(action))
        elif input_format == "symbol" and output_format == "text":
            converted.append(converter.convert_symbol_to_text(action))
        elif input_format == "text" and output_format == "index":
            converted.append(str(converter.convert_text_to_index(action)))
        elif input_format == "index" and output_format == "text":
            converted.append(converter.convert_index_to_text(int(action)))
        elif input_format == "symbol" and output_format == "index":
            converted.append(str(converter.convert_symbol_to_index(action)))
        elif input_format == "index" and output_format == "symbol":
            converted.append(converter.convert_index_to_symbol(int(action)))
        elif input_format == "binary" and output_format == "text":
            converted.append(converter.convert_binary_to_text(int(action)))
        elif input_format == "text" and output_format == "binary":
            converted.append(str(converter.convert_text_to_binary(action)))
        elif input_format == "binary" and output_format == "symbol":
            converted.append(converter.convert_binary_to_symbol(int(action)))
        elif input_format == "binary" and output_format == "index":
            converted.append(str(converter.convert_binary_to_index(int(action))))
        else:
            # No conversion needed or unsupported format combination
            converted.append(action)

    return converted


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert N++ action sequences between different formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert text actions to symbols
  python -m nclone.replay.convert_actions --input actions.txt --output symbols.txt --input-format text --output-format symbol
  
  # Convert symbols to discrete indices
  python -m nclone.replay.convert_actions --input symbols.txt --output indices.txt --input-format symbol --output-format index
  
  # Convert comma-separated actions
  python -m nclone.replay.convert_actions --input "NOOP,Jump,Right" --output-format symbol --separator ","
        """,
    )

    parser.add_argument(
        "--input", type=str, required=True, help="Input file path or action string"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (prints to stdout if not specified)",
    )
    parser.add_argument(
        "--input-format",
        choices=["text", "symbol", "index", "binary"],
        default="text",
        help="Input format (default: text)",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "symbol", "index", "binary"],
        default="symbol",
        help="Output format (default: symbol)",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default=None,
        help="Action separator (comma, space, etc.). If not specified, treats input as file path.",
    )
    parser.add_argument(
        "--join-output",
        type=str,
        default="",
        help="String to join output actions (default: empty string for symbols, comma for others)",
    )

    args = parser.parse_args()

    # Determine if input is a file path or a string
    if args.separator:
        # Input is an action string
        actions = args.input.split(args.separator)
        actions = [action.strip() for action in actions]
    else:
        # Input is a file path
        input_path = Path(args.input)
        if not input_path.exists():
            parser.error(f"Input file does not exist: {args.input}")

        with open(input_path, "r") as f:
            content = f.read().strip()

        # Try to detect format
        if "," in content:
            actions = [action.strip() for action in content.split(",")]
        elif " " in content:
            actions = content.split()
        else:
            # Assume each character is an action (for symbol format)
            actions = list(content)

    # Convert actions
    converted_actions = convert_action_sequence(
        actions, args.input_format, args.output_format
    )

    # Determine output join string
    join_str = args.join_output
    if not join_str:
        if args.output_format == "symbol":
            join_str = ""  # Symbols are typically concatenated
        else:
            join_str = ","  # Text and indices are typically comma-separated

    # Create output string
    output_str = join_str.join(converted_actions)

    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(output_str)

        print(
            f"Converted {len(actions)} actions from {args.input_format} to {args.output_format}"
        )
        print(f"Output saved to: {args.output}")
    else:
        print(output_str)


if __name__ == "__main__":
    main()
