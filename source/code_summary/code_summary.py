"""
Code File Merger Tool.

This tool merges multiple code files into a single text report,
facilitating code review, documentation generation, or submission to AI assistants for analysis.
Supports direct file path input or batch reading from a list file,
with automatic filtering of binary files and invalid paths.

Features:
    1. Smart File Filtering: Auto-detects and skips binary files, directories, and non-existent paths
    2. Dual Input Modes: Supports direct CLI arguments or reading from a text list file
    3. Structured Output: Numbered file separators and clear header statistics
    4. Error Handling: Single file read failure does not affect the overall process
    5. Path Safety: Auto-resolves to absolute paths to avoid relative path confusion

Usage:
    # Method 1: Direct arguments
    python -m source.code_summary.code_summary ./src/main.py ./utils/helper.py ./config.yaml

    # Method 2: Read from list file (supports comment lines with #)
    python -m source.code_summary.code_summary --list source\\code_summary\\code_list.txt
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILENAME = SCRIPT_DIR / "code_summary.txt"
SEPARATOR_LINE = "=" * 80


def is_binary_file(file_path: Path) -> bool:
    """Heuristically detect binary files by checking for null bytes."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\x00' in chunk
    except Exception:
        return True


def process_files(file_paths: list, output_file: Path):
    """Merge a list of files into a single structured text report."""
    valid_files = []

    print(f"Checking {len(file_paths)} file paths...")

    for path_str in file_paths:
        path = Path(path_str.strip())

        if not path.exists():
            print(f"   Skipped (not found): {path}")
            continue

        if path.is_dir():
            print(f"   Skipped (is directory): {path}")
            continue

        if is_binary_file(path):
            print(f"   Skipped (binary file): {path}")
            continue

        valid_files.append(path)

    if not valid_files:
        print("No valid code files found.")
        return

    print(f"Found {len(valid_files)} valid files, generating {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("Code Merge Report\n")
        outfile.write(f"Total Files: {len(valid_files)}\n")
        outfile.write(f"{SEPARATOR_LINE}\n\n")

        for i, file_path in enumerate(valid_files, 1):
            outfile.write(f"### File {i}/{len(valid_files)}: {file_path} ###\n")
            outfile.write("-" * 40 + "\n")

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                    content = infile.read()
                    outfile.write(content)
            except Exception as e:
                outfile.write(f"[Read Error: {e}]")

            outfile.write(f"\n\n{SEPARATOR_LINE}\n\n")

    print(f"Done! All code saved to: {output_file.resolve()}")


def main():
    """Command-line entry point."""
    args = sys.argv[1:]
    file_list = []

    if not args:
        print("Usage:")
        print("  1. Direct paths: python -m source.code_summary.code_summary ./a.py ./b.py")
        print("  2. List file:    python -m source.code_summary.code_summary --list paths.txt")
        return

    if args[0] == "--list":
        if len(args) < 2:
            print("Error: Please specify a file containing the path list")
            return

        list_file = Path(args[1])

        if list_file.exists():
            with open(list_file, 'r', encoding='utf-8') as f:
                file_list = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith('#')
                ]
        else:
            print(f"Error: List file {list_file} does not exist")
            return
    else:
        file_list = args

    process_files(file_list, OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
