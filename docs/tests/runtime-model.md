# Batch 2: runtime with a model

Spec version under test: **2.0**. Levels: see [runtime-static-lifecycle.md](runtime-static-lifecycle.md#levels).
These tests need a model file the runtime can load: `oaax-conformance runtime LIB --model FILE`. Without `--model` they are `BLOCKED`;
if the runtime cannot load the given model they are `BLOCKED` too, not failed.

## `RT-M`: models

| ID | Check | Pass when | Level | Source | Probed | Status |
| -- | ----- | --------- | ----- | ------ | ------ | ------ |
| RT-M-01 | Second `load_models` | after a successful `load_models`, another call returns an error status (no crash); the runtime reports `ALREADY_INITIALIZED` today | PROPOSED | none: the header says loaded models are maintained "until runtime_cleanup()" but not what a later call does; models are replaced only by `cleanup` + `init` | 2/2 with an ONNX model (reference CPU, PythonBindings CPU: `ALREADY_INITIALIZED`); Intel and Hailo need their own model formats: not probed | approved |
