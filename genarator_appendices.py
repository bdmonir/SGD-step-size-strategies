#!/usr/bin/env python3
"""
Generate LaTeX appendix files from Python source code.

This script scans the current directory for Python files and creates
a corresponding LaTeX file for each source file. The generated files
use the minted package for syntax highlighting and are collected in
a main appendices.tex file that can be included in the thesis.
"""

from pathlib import Path


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SOURCE_DIRECTORY = Path(".")
APPENDIX_DIRECTORY = Path("appendices")
MAIN_APPENDIX_FILE = APPENDIX_DIRECTORY / "appendices.tex"


def escape_latex(text: str) -> str:
    """
    Escape characters that have a special meaning in LaTeX.

    Parameters
    ----------
    text : str
        Input text to be escaped.

    Returns
    -------
    str
        LaTeX-safe text.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }

    for character, replacement in replacements.items():
        text = text.replace(character, replacement)

    return text


def generate_appendix_file(source_file: Path) -> Path:
    """
    Generate a LaTeX appendix file for a Python source file.

    Parameters
    ----------
    source_file : Path
        Path to the Python source file.

    Returns
    -------
    Path
        Path to the generated LaTeX file.
    """
    appendix_file = APPENDIX_DIRECTORY / f"{source_file.stem}.tex"
    chapter_title = escape_latex(source_file.stem.replace("_", " ").title())

    source_code = source_file.read_text(encoding="utf-8")

    latex_content = (
        f"\\chapter{{{chapter_title}}}\n"
        f"\\label{{appendix:{source_file.stem}}}\n\n"
        "\\begin{minted}[\n"
        "    frame=lines,\n"
        "    framesep=2mm,\n"
        "    fontsize=\\footnotesize,\n"
        "    linenos,\n"
        "    breaklines,\n"
        "    breakanywhere\n"
        "]{python}\n"
        f"{source_code}\n"
        "\\end{minted}\n"
    )

    appendix_file.write_text(latex_content, encoding="utf-8")

    return appendix_file


def generate_appendices() -> None:
    """
    Generate LaTeX appendix files for all Python files in the
    configured source directory.
    """
    APPENDIX_DIRECTORY.mkdir(parents=True, exist_ok=True)

    python_files = sorted(SOURCE_DIRECTORY.glob("*.py"))

    # Exclude this generator script from the thesis appendix.
    python_files = [
        file for file in python_files
        if file.name != Path(__file__).name
    ]

    appendix_entries = [
        "% Automatically generated appendix file.\n",
        "% Include this file in the main thesis document.\n\n",
    ]

    for source_file in python_files:
        appendix_file = generate_appendix_file(source_file)

        print(
            f"Generated appendix: "
            f"{source_file.name} -> {appendix_file}"
        )

        appendix_entries.append(
            f"\\input{{{appendix_file.as_posix()}}}\n"
        )

    MAIN_APPENDIX_FILE.write_text(
        "".join(appendix_entries),
        encoding="utf-8",
    )

    print(
        f"\nAppendix generation completed. "
        f"Main file: {MAIN_APPENDIX_FILE}"
    )


if __name__ == "__main__":
    generate_appendices()