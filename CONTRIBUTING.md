# Contributing to the OAAX conformance suite

Please read the [OAAX general CONTRIBUTING document](https://github.com/OAAX-standard/OAAX/blob/main/CONTRIBUTING.md)
first: it sets out the guidelines for the whole project, including the sign-off every commit needs (`git commit -s`).

## How to contribute

1. Open an issue to discuss the test or fix you have in mind.
2. Fork the repository and create a branch from `main`.
3. Add or change tests. Each test names the specification clause it checks and its level (MUST, SHOULD, PROPOSED).
4. Commit with a clear, signed-off message and open a pull request against `main`.

## Development environment

```bash
scripts/build.sh && uv sync && uv run pytest -n auto
```

To add a test: write it in `docs/tests/` first (level, source clause, pass condition) and get it `approved`, then implement the check in
`python/oaax_conformance/checks/` with `@check("<ID>")` and add a fault to `mock-runtime/mock.c` plus a case in `tests/test_suite.py`
that makes it fail. The self-tests insist on all three.

The toolchain self-tests build small mock toolchain images (`mock-toolchain/`) and need Docker; without it they are skipped. To build them from an
image you already have instead of pulling `python:3.11-slim`, set `OAAX_MOCK_BASE` to any image that has python3 (for example the CPU toolchain).
For a new toolchain test, add a fault to `mock-toolchain/mock_toolchain.py` and a case to `tests/test_toolchain.py`.

## Releasing (maintainers)

Set `__version__` in `python/oaax_conformance/__init__.py` (the one place) and move the `CHANGELOG.md` entry from "unreleased" to the version and a date (`## 0.1.0 - 2026-11-03`). Push a tag `vX.Y.Z` on `main`:
the `build` workflow checks that the tag is a commit of `main` and matches the version, that the CHANGELOG heading is dated, builds and smoke-tests the wheel (specs inside, exit codes), and creates a **draft** GitHub Release with the wheel, the checksums and the CHANGELOG notes for a maintainer to review and publish. A re-run refuses if a release for the tag exists. Nothing goes to PyPI.
