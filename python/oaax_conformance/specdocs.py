"""The test specifications in docs/tests/*.md are the single source of the id, title, level, source and status of every check."""
from dataclasses import dataclass
from pathlib import Path

COLUMNS = ["id", "title", "pass_when", "level", "source", "probed", "status"]


@dataclass(frozen=True)
class Spec:
    id: str
    title: str
    pass_when: str
    level: str
    source: str
    status: str


def spec_dir() -> Path:
    packaged = Path(__file__).parent / "test_specs"  # wheel: docs/tests is force-included here
    return packaged if packaged.is_dir() else Path(__file__).resolve().parents[2] / "docs" / "tests"


ORDER = {"RT-S": 0, "RT-L": 1, "RT-M": 2, "TC": 3, "E2E": 4}


def sort_key(test_id: str) -> tuple:
    """RT-S before RT-L before RT-M, then TC and E2E; numbers as numbers."""
    prefix, _, number = test_id.rpartition("-")
    return (ORDER.get(prefix, 9), int(number) if number.isdigit() else 0)


def load(directory: Path | None = None) -> dict[str, Spec]:
    specs = {}
    for path in sorted((directory or spec_dir()).glob("*.md")):
        for line in path.read_text().splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not line.startswith("| ") or not cells[0].startswith(("RT-", "TC-", "E2E-")):
                continue
            row = dict(zip(COLUMNS, cells))
            assert len(cells) == len(COLUMNS), f"{path.name}: {cells[0]} has {len(cells)} cells, expected {len(COLUMNS)}"
            specs[row["id"]] = Spec(row["id"], row["title"], row["pass_when"], row["level"], row["source"], row["status"])
    return dict(sorted(specs.items(), key=lambda item: sort_key(item[0])))
