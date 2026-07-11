"""
Code File Merger Tool.

This tool merges multiple code files into a single text report,
facilitating code review, documentation generation, or submission to AI assistants for analysis.
Supports direct file path input, batch reading from a list file, or auto-discovery.

Usage (from the repository root):
    # Method 1: Auto-discover all .py files in the current directory (default)
    python tools/code_summary.py

    # Method 2: Direct arguments
    python tools/code_summary.py ./source/geometry.py ./source/assets.py

    # Method 3: Read from list file (supports comment lines with #)
    python tools/code_summary.py --list code_list.txt
"""

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
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge multiple code files into a single text report."
    )
    parser.add_argument(
        "files", 
        nargs="*", 
        help="Direct file paths to merge. If omitted, auto-discovers .py files in the current directory."
    )
    parser.add_argument(
        "--list", 
        dest="list_file", 
        help="Read file paths from a text list file (supports # comments)."
    )

    args = parser.parse_args()
    file_list = []

    # Mode 1: Read from a list file
    if args.list_file:
        list_path = Path(args.list_file)
        if list_path.exists():
            with open(list_path, 'r', encoding='utf-8') as f:
                file_list = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith('#')
                ]
        else:
            print(f"Error: List file {list_path} does not exist")
            return

    # Mode 2: Direct arguments provided
    elif args.files:
        file_list = args.files

    # Mode 3: Default mode - Auto-discover .py files in the current running directory
    else:
        current_dir = Path.cwd()
        print(f"No arguments provided. Auto-discovering .py files in: {current_dir}")
        # rglob("*") 会递归搜索当前目录及所有子目录下的 .py 文件
        discovered_files = sorted([str(p) for p in current_dir.rglob("*.py")])
        
        if not discovered_files:
            print("No .py files found in the current directory.")
            return
            
        file_list = discovered_files

    process_files(file_list, OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
