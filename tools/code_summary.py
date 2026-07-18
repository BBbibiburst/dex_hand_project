"""
Code File Merger Tool.

This tool merges multiple code files into a single text report,
facilitating code review, documentation generation, or submission to AI assistants for analysis.
Supports direct file path input, batch reading from a list file, or auto-discovery.

Usage (from the repository root):
    # Method 1: Auto-discover files matching DEFAULT_EXTENSIONS
    python tools/code_summary.py

    # Method 2: Auto-discover selected file types
    python tools/code_summary.py --extensions py js ts json
    python tools/code_summary.py --extensions .py,.md,.yaml

    # Method 3: Direct arguments
    python tools/code_summary.py ./source/geometry.py ./source/assets.py

    # Method 4: Read from list file (supports comment lines with #)
    python tools/code_summary.py --list code_list.txt
"""

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILENAME = SCRIPT_DIR / "code_summary.txt"
SEPARATOR_LINE = "=" * 80

# File types used by auto-discovery when --extensions is not provided.
# Add or remove extensions here to customize the project default.
DEFAULT_EXTENSIONS = (
    ".py",
    ".xml",
    ".json",
)


def normalize_extensions(values: list[str]) -> tuple[str, ...]:
    """Normalize extension arguments such as 'py', '.py', or 'py,js'."""
    extensions = []
    for value in values:
        for item in value.split(","):
            item = item.strip().lower()
            if not item:
                continue
            extension = item if item.startswith(".") else f".{item}"
            if extension not in extensions:
                extensions.append(extension)

    if not extensions:
        raise ValueError("At least one non-empty file extension is required.")

    return tuple(extensions)


def is_binary_file(file_path: Path) -> bool:
    """Heuristically detect binary files by checking for null bytes."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            return b"\x00" in chunk
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

    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write("Code Merge Report\n")
        outfile.write(f"Total Files: {len(valid_files)}\n")
        outfile.write(f"{SEPARATOR_LINE}\n\n")

        for i, file_path in enumerate(valid_files, 1):
            outfile.write(f"### File {i}/{len(valid_files)}: {file_path} ###\n")
            outfile.write("-" * 40 + "\n")

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
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
        help="Direct file paths to merge. If omitted, auto-discovers configured file types.",
    )
    parser.add_argument(
        "--list",
        dest="list_file",
        help="Read file paths from a text list file (supports # comments).",
    )
    parser.add_argument(
        "-e",
        "--extensions",
        nargs="+",
        metavar="EXT",
        help=(
            "File types for auto-discovery, separated by spaces or commas "
            f"(default: {' '.join(DEFAULT_EXTENSIONS)})."
        ),
    )

    args = parser.parse_args()
    file_list = []

    # Mode 1: Read from a list file
    if args.list_file:
        list_path = Path(args.list_file)
        if list_path.exists():
            with open(list_path, "r", encoding="utf-8") as f:
                file_list = [
                    line.strip() for line in f if line.strip() and not line.startswith("#")
                ]
        else:
            print(f"Error: List file {list_path} does not exist")
            return

    # Mode 2: Direct arguments provided
    elif args.files:
        file_list = args.files

    # Mode 3: Auto-discover configured file types in the current directory
    else:
        current_dir = Path.cwd()
        try:
            extensions = normalize_extensions(args.extensions or list(DEFAULT_EXTENSIONS))
        except ValueError as exc:
            parser.error(str(exc))

        print(
            f"No file paths provided. Auto-discovering "
            f"{', '.join(extensions)} files in: {current_dir}"
        )
        # Match suffixes in one recursive traversal so multiple types are supported.
        discovered_files = sorted(
            str(path)
            for path in current_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )

        if not discovered_files:
            print(f"No files matching {', '.join(extensions)} were found.")
            return

        file_list = discovered_files

    process_files(file_list, OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
