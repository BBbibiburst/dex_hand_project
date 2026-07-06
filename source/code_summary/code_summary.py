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

Output Format:
    Code Merge Report
    Total Files: N
    ================================================================================
    
    ### File 1/N: /path/to/file1.py ###
    ----------------------------------------
    [File 1 Content]
    
    ================================================================================
    
    ### File 2/N: /path/to/file2.py ###
    ----------------------------------------
    [File 2 Content]

Usage:
    # Method 1: Direct arguments
    python -m source.code_summary.code_summary ./src/main.py ./utils/helper.py ./config.yaml
    
    # Method 2: Read from list file (supports comment lines with #)
    python -m source.code_summary.code_summary --list source\code_summary\code_list.txt
    
    # file_list.txt example:
    # This is a comment line, will be ignored
    ./src/main.py
    ./utils/helper.py
    ./README.md
"""

import sys
import os
from pathlib import Path

# ====================== Configuration ======================

# Get the absolute path of the current script file, and use its parent directory as the base path
SCRIPT_DIR = Path(__file__).resolve().parent
# Output file path: code_summary.txt in the same directory as the script
OUTPUT_FILENAME = SCRIPT_DIR / "code_summary.txt"

# File separator line style (80 equals signs)
SEPARATOR_LINE = "=" * 80


# ====================== Internal Helper Functions ======================

def is_binary_file(file_path: Path) -> bool:
    """
    Detects whether a file is a binary file.

    Algorithm: Reads the first 1024 bytes of the file and checks for null bytes (\x00).
    Text files typically do not contain null bytes, while binary files 
    (images, executables, etc.) usually do.

    Args:
        file_path: The file path to be checked.

    Returns:
        bool: True if null bytes are detected (classified as binary), False otherwise.
            Returns True on read failure as well (conservative strategy to avoid processing corrupted files).

    Note:
        This is a heuristic detection. It may misclassify certain special encoded text files,
        but it is sufficiently reliable for common code files (UTF-8, ASCII).
    """
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\x00' in chunk
    except Exception:
        # Conservative approach: treat read failure as binary (or corrupted) and skip processing
        return True


def process_files(file_paths: list, output_file: Path):
    """
    Processes a list of files and merges them into a single output file.

    Processing flow:
        1. Path validation: checks existence, file type, and binary detection
        2. Valid file statistics and reporting
        3. Sequential reading and writing with structured separator markers
        4. Error isolation: single file failure does not affect other files

    Args:
        file_paths: A list of input file path strings.
        output_file: A Path object for the output file.

    Returns:
        None. Results are written directly to the disk file.

    Side Effects:
        Creates/overwrites the file specified by output_file.
        Prints processing progress to standard output.

    Examples:
        >>> paths = ["./test.py", "./README.md"]
        >>> process_files(paths, Path("./output.txt"))
        🔍 Checking 2 file paths...
        ✅ Found 2 valid files, generating output.txt...
        🎉 Done! All code saved to: /absolute/path/to/output.txt
    """
    valid_files = []
    
    print(f"🔍 Checking {len(file_paths)} file paths...")
    
    # ----- Phase 1: Path Validation and Filtering -----
    for path_str in file_paths:
        # Clean input: strip leading/trailing whitespace (handles spaces from copy-paste)
        path = Path(path_str.strip())
        
        # 1. Check if path exists (file or directory)
        if not path.exists():
            print(f"   ⚠️ Skipped (not found): {path}")
            continue
            
        # 2. Check if it is a file (exclude directories)
        if path.is_dir():
            print(f"   ⚠️ Skipped (is directory): {path}")
            continue
            
        # 3. Check if it is a binary file (avoid mixing images, etc. into text report)
        if is_binary_file(path):
            print(f"   ⚠️ Skipped (binary file): {path}")
            continue
            
        valid_files.append(path)

    # Early exit if no valid files found
    if not valid_files:
        print("❌ No valid code files found.")
        return

    # ----- Phase 2: Merge and Write -----
    print(f"✅ Found {len(valid_files)} valid files, generating {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Write report header information
        outfile.write(f"Code Merge Report\n")
        outfile.write(f"Total Files: {len(valid_files)}\n")
        outfile.write(f"{SEPARATOR_LINE}\n\n")

        # Process valid files one by one
        for i, file_path in enumerate(valid_files, 1):
            # Write file marker header (includes sequence number and absolute path)
            outfile.write(f"### File {i}/{len(valid_files)}: {file_path} ###\n")
            outfile.write("-" * 40 + "\n")
            
            try:
                # Read file content (ignore encoding errors to ensure process continuity)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                    content = infile.read()
                    outfile.write(content)
            except Exception as e:
                # Single file error does not affect the whole, record error info and continue
                outfile.write(f"[Read Error: {e}]")
            
            # Write separator (blank line + separator line + blank line)
            outfile.write(f"\n\n{SEPARATOR_LINE}\n\n")

    # Inform user of the absolute file path for easy lookup
    print(f"🎉 Done! All code saved to: {output_file.resolve()}")


# ====================== Main Entry Point ======================

def main():
    """
    Command-line entry function, handles argument parsing and flow control.

    Supported command-line modes:
        1. Direct argument mode: python -m source.code_summary.code_summary <file1> <file2> ...
        2. List file mode: python -m source.code_summary.code_summary --list <path_to_list_file>

    List file format:
        - One file path per line
        - Lines starting with # are comments and will be ignored
        - Empty lines are automatically skipped

    Args:
        Receives command-line arguments via sys.argv.

    Returns:
        None. Calls process_files based on arguments or prints help information directly.

    Examples:
        $ python -m source.code_summary.code_summary ./src/main.py ./utils/helper.py ./config.yaml
        $ python -m source.code_summary.code_summary --list ./source/tools/path.txt
        $ python -m source.code_summary.code_summary --list ./source/tools/path.txt > output.txt
        Usage:
          1. Pass file paths directly: python -m source.code_summary.code_summary ./src/main.py ./utils/helper.py ./config.yaml
          2. Read from list file:     python -m source.code_summary.code_summary --list paths.txt
    """
    # Get command-line arguments (excluding the script name itself)
    args = sys.argv[1:]
    
    file_list = []

    # Show help when no arguments provided
    if not args:
        print("Usage:")
        print("  1. Pass file paths directly: python -m source.code_summary.code_summary ./src/main.py ./utils/helper.py ./config.yaml")
        print("  2. Read from list file:     python -m source.code_summary.code_summary --list paths.txt")
        return

    # ----- Mode 1: Read from list file -----
    if args[0] == "--list":
        if len(args) < 2:
            print("Error: Please specify a file containing the path list")
            return
        
        list_file = Path(args[1])
        
        if list_file.exists():
            with open(list_file, 'r', encoding='utf-8') as f:
                # Filter: non-empty lines and non-comment lines (starting with #)
                file_list = [
                    line.strip() 
                    for line in f 
                    if line.strip() and not line.startswith('#')
                ]
        else:
            print(f"Error: List file {list_file} does not exist")
            return
    
    # ----- Mode 2: Direct arguments -----
    else:
        file_list = args

    # Execute core processing logic
    process_files(file_list, OUTPUT_FILENAME)


if __name__ == "__main__":
    main()