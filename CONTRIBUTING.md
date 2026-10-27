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

### Adding a test

1. Write it in `docs/tests/` first (level, source clause, pass condition) and get it `approved`.
2. Implement the check in `python/oaax_conformance/checks/` with `@check("<ID>")`.
3. Add a fault to `mock-runtime/mock.c` (or `mock-toolchain/mock_toolchain.py`) that makes the check fail.
4. Add a case in `tests/test_suite.py` (or `tests/test_toolchain.py`) that proves it.

The self-tests insist on all of it.

### Toolchain self-tests

They build small mock toolchain images (`mock-toolchain/`) and need Docker; without it they are skipped.
To build them from an image you already have instead of pulling `python:3.11-slim`, set `OAAX_MOCK_BASE` to any image that has python3 (for example the CPU toolchain).

## Releasing (maintainers)

1. Set `__version__` in `python/oaax_conformance/__init__.py` (the one place).
2. In `CHANGELOG.md`, move the entry from "unreleased" to the version and a date (`## 0.1.0 - 2026-11-03`).
3. Push a tag `vX.Y.Z` on `main`.

The `build` workflow then:

- checks that the tag is a commit of `main`, matches the version, and that the CHANGELOG heading is dated;
- builds and smoke-tests the wheel (specs inside, exit codes);
- creates a **draft** GitHub Release with the wheel, the checksums and the CHANGELOG notes, for a maintainer to review and publish.

A re-run refuses if a release for the tag exists. Nothing goes to PyPI.
