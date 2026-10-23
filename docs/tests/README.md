# Test specifications

The suite is specified here before it is implemented. **Only tests marked `approved` get implemented.**

## How to read a test

| Column | Meaning |
| ------ | ------- |
| ID | `RT-<group>-<nn>` runtime, `TC-` toolchain, `E2E-` end to end. IDs never change or get reused. |
| Level | **MUST**: required by the text of the spec version under test; failing it fails the run. **SHOULD**: recommended by the text, or derived from it; a failure is a warning. **PROPOSED**: not in the text; the behaviour planned for the next version; always a warning. |
| Source | The clause the test comes from (`interface.h` = the normative header). A test without a quote from the text is never MUST. |
| Probed | Result of the same check run by hand on the runtimes we have, before any implementation. |
| Status | `draft` (under review), `approved` (implement it), `dropped`. |

## Review

Comment on the pull request, or change the Status/Level column directly. A test's level is the thing most worth arguing about:
a MUST that the spec text does not support makes the suite wrong, not the vendor.

## Files

- [`runtime-static-lifecycle.md`](runtime-static-lifecycle.md): batch 1, runtime without a model.
- [`runtime-model.md`](runtime-model.md): batch 2, tests that need a model (`--model`).
