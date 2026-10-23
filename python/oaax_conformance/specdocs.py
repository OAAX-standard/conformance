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
    return specs
