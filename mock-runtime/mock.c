// A tiny OAAX 2.0 runtime for testing the conformance suite itself. It is correct unless the environment variable
// OAAX_MOCK_FAULT names a fault: every check of the suite must fail on exactly its fault and pass on the clean mock.
// No models: load/enqueue always fail once initialized. Build variants (see CMakeLists.txt) cover missing/extra exports.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "interface.h"

#define EXPORT __attribute__((visibility("default")))

static int initialized, cleaned_once, has_error, models_loaded;
static char error_text[160];

static int fault(const char *name) {
  const char *f = getenv("OAAX_MOCK_FAULT");
  return f && !strcmp(f, name);
}

static RuntimeStatus fail(RuntimeStatus status, const char *message) {
  if (!fault("no_error_text")) {
    strncpy(error_text, message, sizeof error_text - 1);
    has_error = 1;
  }
  return status;
}

static void *idle_thread(void *arg) {
  (void)arg;
  pthread_setname_np(pthread_self(), "mock-worker");
  for (;;) pause();
  return NULL;
}

// A call while not initialized (before init, or after cleanup): NOT_INITIALIZED, unless a fault says otherwise.
static RuntimeStatus not_initialized(void) {
  if (fault("preinit_success") || (cleaned_once && fault("postcleanup_success"))) return RUNTIME_STATUS_SUCCESS;
  if (fault("crash_preinit")) *(volatile int *)0 = 1;
  return fail(RUNTIME_STATUS_NOT_INITIALIZED, "runtime not initialized");
}

EXPORT RuntimeStatus runtime_init(Config config) {
  if (initialized) {
    if (fault("init_twice_ok")) return RUNTIME_STATUS_SUCCESS;
    return fail(RUNTIME_STATUS_ALREADY_INITIALIZED, "already initialized");
  }
  if (fault("require_config") && config.length == 0) return fail(RUNTIME_STATUS_ERROR, "config required");
  for (int i = 0; i < config.length; i++)
    if (fault("reject_unknown_key") && !strcmp(config.keys[i], "oaax_conformance_unknown_key"))
      return fail(RUNTIME_STATUS_INVALID_ARGUMENT, "unknown key");
  if (cleaned_once && fault("no_reinit")) return fail(RUNTIME_STATUS_ERROR, "cannot re-initialize");

  if (fault("leak_thread")) {
    pthread_t t;
    pthread_create(&t, NULL, idle_thread, NULL);
    pthread_detach(t);
  }
  if (fault("leak_memory")) memset(malloc(1 << 20), 1, 1 << 20);
  if (fault("pin_library")) {
    Dl_info info;
    if (dladdr((void *)runtime_init, &info)) dlopen(info.dli_fname, RTLD_NOW | RTLD_NODELETE);
  }
  initialized = 1;
  if (!fault("stale_error")) has_error = 0;  // a successful call leaves no error behind
  return RUNTIME_STATUS_SUCCESS;
}

// The mock "model" is any file that starts with MOCKMODEL. Loading is once per init: a later call is an error.
// A missing file is FILE_NOT_FOUND, any other file INVALID_MODEL.
EXPORT RuntimeStatus runtime_load_models(int num_models, const ModelConfig *model_configs) {
  if (!initialized) return not_initialized();
  if (num_models < 1 || !model_configs || !model_configs[0].file_path) return fail(RUNTIME_STATUS_INVALID_ARGUMENT, "no model given");
  if (models_loaded && !fault("load_twice_ok")) return fail(RUNTIME_STATUS_ALREADY_INITIALIZED, "models are already loaded");
  if (fault("load_error_generic")) return fail(RUNTIME_STATUS_ERROR, "failed to load the model");
  FILE *f = fopen(model_configs[0].file_path, "rb");
  if (!f) return fail(RUNTIME_STATUS_FILE_NOT_FOUND, "model file not found");
  char magic[9] = "";
  size_t n = fread(magic, 1, sizeof magic, f);
  fclose(f);
  if (n != sizeof magic || memcmp(magic, "MOCKMODEL", sizeof magic)) return fail(RUNTIME_STATUS_INVALID_MODEL, "not a model");
  models_loaded = 1;
  return RUNTIME_STATUS_SUCCESS;
}

EXPORT RuntimeStatus runtime_enqueue_input(int model_id, Tensors *input_tensors) {
  (void)model_id, (void)input_tensors;
  return initialized ? fail(RUNTIME_STATUS_MODEL_NOT_LOADED, "no model loaded") : not_initialized();
}

EXPORT RuntimeStatus runtime_retrieve_output(int *model_id, Tensors **output_tensors, int timeout_ms) {
  (void)timeout_ms;
  if (fault("null_ptr_crash")) *output_tensors = NULL;  // dereferences the pointer the caller left NULL
  if (!model_id || !output_tensors) {
    if (fault("null_ptr_ignored")) return RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
    return fail(RUNTIME_STATUS_INVALID_ARGUMENT, "model_id and output_tensors must not be NULL");
  }
  return initialized ? RUNTIME_STATUS_NO_OUTPUT_AVAILABLE : not_initialized();
}

EXPORT RuntimeStatus runtime_cleanup(void) {
  if (!initialized && fault("cleanup_not_idempotent")) return fail(RUNTIME_STATUS_ERROR, "not initialized");
  initialized = 0;
  models_loaded = 0;
  cleaned_once = 1;
  return RUNTIME_STATUS_SUCCESS;
}

EXPORT const char *runtime_get_error(void) {
  if (fault("error_at_start")) return "stale error";
  return has_error ? error_text : NULL;
}

EXPORT const char *runtime_get_version(void) { return fault("bad_version") ? "v1" : "1.0.0"; }

EXPORT const char *runtime_get_name(void) {
  static char changing[32];
  static int calls;
  if (fault("null_name")) return NULL;
  if (!fault("unstable_getters")) return "OAAX Mock Runtime";
  snprintf(changing, sizeof changing, "OAAX Mock %d", calls++);
  return changing;
}

#ifndef MOCK_OMIT_INFO
EXPORT const char *runtime_get_info(void) {
  if (initialized) return fault("info_not_json") ? "not json" : "{\"loaded_models\":0}";
  if (fault("info_before_init") || (cleaned_once && fault("info_after_cleanup"))) return "{}";
  return NULL;
}
#endif

#ifdef MOCK_EXTRA_EXPORT
EXPORT int mock_extra_symbol(void) { return 1; }
#endif

__attribute__((destructor)) static void at_exit(void) {
  if (initialized && fault("crash_at_exit")) abort();
}
