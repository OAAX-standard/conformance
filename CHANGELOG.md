# Changelog

## 0.1.0 (unreleased)

First release.

- Runtime tiers (`static`, `lifecycle`, `model`): 62 checks against the C API of an OAAX 2.0 runtime, each with its spec clause and level (MUST / SHOULD / PROPOSED), run by a C harness (one scenario per process).
- Toolchain tier (30 checks) and end-to-end tier (4): a Docker image (arguments, exit codes, log, no root, no network, read-only, hostile input), then convert, load, run and compare with onnxruntime.
- Reports: console table, `report.html` (self-contained, prints to PDF), `report.json`, `junit.xml`; values of `--init-config` and directory names are never stored.
- Safe on real hardware: time-boxed loops, capped request sizes, a slow device is `BLOCKED`, never `FAIL`; `--hardware-safe` skips the abusive tests.
- Exit codes: 0 clean, 1 a MUST failed, 2 incomplete, 3 usage, 4 a check of the suite crashed, 130 / 143 / 129 stopped by Ctrl-C / SIGTERM / SIGHUP (harness and containers are stopped first).
- Hardened by seven adversarial review rounds (the last three: secrets in every `docker` spelling, every `docker run` flag that leaves the foreground refused, a manifest that is no object or names a link, a model too big to read, untrusted files a toolchain leaves behind are read safely, a library that cannot load here is one line, every conversion runs with `--cap-drop ALL`, `--model-file`, a `note` on results that rest on less work than specified, a timing the spec does not bound is a WARN, never a FAIL); every finding has a regression test, and the self-tests inject a fault into a mock runtime and a mock toolchain for every check.
