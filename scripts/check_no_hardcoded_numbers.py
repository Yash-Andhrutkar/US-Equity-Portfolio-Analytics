"""
Anti-fabrication guard for the dashboard front end.

The dashboard's promise is that every financial figure on screen originates
from the Python analytics layer via docs/data/dashboard_data.json. This
script enforces that promise mechanically, so a hand-pasted number cannot
survive a commit.

Three independent checks:

  1. LITERAL VALUES - every figure in output/tables/*.csv is rendered at
     several plausible display precisions (percent, ratio, raw) and searched
     for in the front-end source. Only renderings with at least
     MIN_SIGNIFICANT_DIGITS significant digits are searched, because short
     strings like "0.25" or "1.2" legitimately occur in CSS and easing
     curves. A hit means a real analytics value was typed into the source.

  2. EMBEDDED SERIES - any numeric array literal longer than
     MAX_ARRAY_LITERAL entries in the front-end source is rejected, since
     that is how a price or return series would be smuggled in. Colour
     ramps and small coefficient tables stay under the limit.

  3. SINGLE SOURCE - the payload must be LOADED from exactly one place. The
     check looks for the loader pattern (the filename inside a `new URL(...)`
     or `fetch(...)` call) rather than any mention of the filename, so prose
     and on-page methodology text may name the file freely while a second
     fetch anywhere in the front end fails the build.

Files under docs/assets/vendor/ (third-party library code) and docs/data/
(the generated payload itself) are excluded.

Exit code 0 = clean, 1 = at least one violation.

Usage:
    python scripts/check_no_hardcoded_numbers.py
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FRONTEND_ROOT = PROJECT_ROOT / "docs"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"

EXCLUDED_DIRS = {
    FRONTEND_ROOT / "assets" / "vendor",
    FRONTEND_ROOT / "data",
}

SCANNED_SUFFIXES = {".html", ".js", ".css"}

DATA_REFERENCE = "dashboard_data.json"

# A literal must carry at least this many significant digits before a match
# is treated as a real finding rather than a coincidence.
MIN_SIGNIFICANT_DIGITS = 4

# Numeric array literals longer than this are treated as embedded data.
MAX_ARRAY_LITERAL = 16

NUMERIC_ARRAY_PATTERN = re.compile(
    r"\[\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){%d,}\s*,?\s*\]" % MAX_ARRAY_LITERAL
)

# The payload being loaded, as opposed to merely named: the filename inside a
# URL construction or a fetch call.
DATA_LOADER_PATTERN = re.compile(
    r"(?:new\s+URL|fetch)\s*\(\s*[\"'`][^\"'`]*" + re.escape(DATA_REFERENCE)
)


def scanned_files() -> list[Path]:
    """Collect front-end source files, skipping vendor and generated data."""
    files = []
    for path in sorted(FRONTEND_ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SCANNED_SUFFIXES:
            continue
        if any(excluded in path.parents for excluded in EXCLUDED_DIRS):
            continue
        files.append(path)
    return files


def _significant_digits(text: str) -> int:
    """Count significant digits in a rendered number."""
    digits = text.replace("-", "").replace(".", "").lstrip("0")
    return len(digits.rstrip("0")) if "." in text else len(digits)


def forbidden_literals() -> dict[str, str]:
    """Render every analytics figure at plausible display precisions.

    Returns a mapping of literal string -> human-readable provenance.
    """
    literals: dict[str, str] = {}

    for csv_path in sorted(TABLES_DIR.glob("*.csv")):
        frame = pd.read_csv(csv_path, index_col=0)
        for column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce")
            for index, value in series.items():
                if value is None or not isinstance(value, float) or not math.isfinite(value):
                    continue

                renderings = [
                    f"{value:.2f}",
                    f"{value:.3f}",
                    f"{value:.4f}",
                    f"{value:.6f}",
                    f"{value * 100:.1f}",
                    f"{value * 100:.2f}",
                ]

                for rendering in renderings:
                    stripped = rendering.lstrip("-")
                    if _significant_digits(stripped) < MIN_SIGNIFICANT_DIGITS:
                        continue
                    literals.setdefault(
                        stripped,
                        f"{csv_path.name}:{index}:{column} = {value!r}",
                    )

    return literals


def check_literals(files: list[Path], literals: dict[str, str]) -> list[str]:
    """Search the front end for rendered analytics values."""
    violations = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for literal, provenance in literals.items():
            if literal in text:
                line_number = text[: text.index(literal)].count("\n") + 1
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{line_number}: hardcoded analytics "
                    f"value '{literal}' (matches {provenance})"
                )
    return violations


def check_embedded_series(files: list[Path]) -> list[str]:
    """Reject long numeric array literals in the front end."""
    violations = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for match in NUMERIC_ARRAY_PATTERN.finditer(text):
            line_number = text[: match.start()].count("\n") + 1
            preview = match.group(0)[:60].replace("\n", " ")
            violations.append(
                f"{path.relative_to(PROJECT_ROOT)}:{line_number}: numeric array literal with "
                f"more than {MAX_ARRAY_LITERAL} entries - series data must come from "
                f"{DATA_REFERENCE} ({preview}...)"
            )
    return violations


def check_single_data_source(files: list[Path]) -> list[str]:
    """Require exactly one place that loads the generated payload."""
    loaders = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        found = len(DATA_LOADER_PATTERN.findall(text))
        if found:
            loaders.append((path, found))

    total = sum(found for _, found in loaders)

    if total == 0:
        return [
            f"no front-end file loads {DATA_REFERENCE} - the dashboard is not "
            f"reading the generated payload"
        ]
    if total > 1:
        detail = ", ".join(
            f"{path.relative_to(PROJECT_ROOT)} (x{found})" for path, found in loaders
        )
        return [
            f"{DATA_REFERENCE} is loaded from {total} places ({detail}) - keep a "
            f"single data entry point so there is one place to audit"
        ]
    return []


def main() -> None:
    header = "Front-end data-provenance guard"
    print(header)
    print("-" * len(header))

    if not FRONTEND_ROOT.exists():
        print(f"::error::{FRONTEND_ROOT.relative_to(PROJECT_ROOT)} does not exist")
        raise SystemExit(1)

    files = scanned_files()
    if not files:
        print("::error::no front-end source files found to scan")
        raise SystemExit(1)

    literals = forbidden_literals()

    violations = (
        check_literals(files, literals)
        + check_embedded_series(files)
        + check_single_data_source(files)
    )

    print(f"Scanned {len(files)} file(s) against {len(literals)} analytics literal(s)")

    if violations:
        print(f"\nFAILED - {len(violations)} violation(s):\n")
        for violation in violations:
            print(f"  ::error::{violation}")
        raise SystemExit(1)

    print("PASSED - no hardcoded financial values, no embedded series, single data source")


if __name__ == "__main__":
    main()
