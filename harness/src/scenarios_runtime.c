// Scenarios of the runtime tiers that need no model (static, lifecycle) and the first model scenario (load_twice).
#include "common.h"
#include "scenarios.h"

// ---- scenarios ----------------------------------------------------------------------------------------------

// No init: getters, info/error before init, cleanup, calls before init.
int scenario_static(void) {
  do_string("get_error_fresh", rt.get_error);  // must be the first runtime call after loading
  do_string("version", rt.get_version);
  do_string("version_again", rt.get_version);
  do_string("name", rt.get_name);
  do_string("name_again", rt.get_name);
  do_string("info_before_init", rt.get_info);
  do_cleanup("cleanup_before_init");
  do_cleanup("cleanup_before_init_again");

  ModelConfig model = {.file_path = "/nonexistent/model.onnx"};
  begin("preinit_load_models");
  done("preinit_load_models", rt.load_models(1, &model), NULL, NULL);

  Tensors *input = make_input();
  begin("preinit_enqueue_input");
  RuntimeStatus s = rt.enqueue_input(0, input);
  done("preinit_enqueue_input", s, NULL, NULL);
  if (s != RUNTIME_STATUS_SUCCESS) free_tensors(input);  // a rejected input stays the caller's

  int model_id = -1;
  Tensors *output = NULL;
  begin("preinit_retrieve_output");
  done("preinit_retrieve_output", rt.retrieve_output(&model_id, &output, 0), NULL, NULL);
  free_tensors(output);
  return 0;
}

// init (unknown key, empty, user config), second init, info/error around it, cleanup, threads.
int scenario_lifecycle(void) {
  char names[8192], extra[8300];
  int baseline = thread_count();
  thread_names(names, sizeof names);
  snprintf(extra, sizeof extra, "\"threads\":%s", names);
  done("threads_baseline", NO_STATUS, NULL, extra);

  if (do_init("init_unknown_key", user_config_plus("oaax_conformance_unknown_key", "1"), 0) == RUNTIME_STATUS_SUCCESS)
    do_cleanup("cleanup_after_unknown_key");
  Config empty = {0};
  if (do_init("init_empty", empty, 0) == RUNTIME_STATUS_SUCCESS) do_cleanup("cleanup_after_empty");

  do_init("init", opt.config, 1);
  do_string("info", rt.get_info);
  do_init("init_second", opt.config, 0);
  do_string("info_after_second_init", rt.get_info);
  do_cleanup("cleanup");
  do_string("info_after_cleanup", rt.get_info);

  settle(baseline, opt.settle_ms);
  thread_names(names, sizeof names);
  snprintf(extra, sizeof extra, "\"threads\":%s", names);
  done("threads_after_cleanup", NO_STATUS, NULL, extra);
  return 0;
}

// --cycles x (init, cleanup): failures, memory and threads.
int scenario_cycles(void) {
  do_init("init", opt.config, 1);  // a runtime that cannot start at all is "blocked", not a re-init failure
  do_cleanup("cleanup_first");
  int baseline = thread_count(), init_failures = 0, cleanup_failures = 0;
  long rss_first = -1;
  char first_failure[128] = "";
  for (int i = 0; i < opt.cycles; i++) {
    begin("cycle");
    RuntimeStatus si = rt.init(opt.config), sc = si == RUNTIME_STATUS_SUCCESS ? rt.cleanup() : RUNTIME_STATUS_ERROR;
    init_failures += si != RUNTIME_STATUS_SUCCESS;
    cleanup_failures += sc != RUNTIME_STATUS_SUCCESS;
    if ((si || sc) && !*first_failure) snprintf(first_failure, sizeof first_failure, "cycle %d: init=%d cleanup=%d", i, si, sc);
    if (i == 0) rss_first = rss_bytes();
  }
  settle(baseline, opt.settle_ms);
  char extra[400];
  snprintf(extra, sizeof extra,
           "\"cycles\":%d,\"init_failures\":%d,\"cleanup_failures\":%d,\"first_failure\":\"%s\",\"rss_first\":%ld,"
           "\"rss_last\":%ld,\"threads_before\":%d,\"threads_after\":%d",
           opt.cycles, init_failures, cleanup_failures, first_failure, rss_first, rss_bytes(), baseline, thread_count());
  done("cycles", NO_STATUS, NULL, extra);
  return 0;
}

// init, cleanup, dlclose: is the library really gone from the process?
int scenario_unload(void) {
  char *resolved = realpath(opt.lib, NULL);
  do_init("init", opt.config, 1);
  do_cleanup("cleanup");
  begin("dlclose");
  int rc = dlclose(lib_handle);
  int mapped = 0;
  char line[PATH_MAX + 256];
  FILE *maps = fopen("/proc/self/maps", "r");
  while (maps && resolved && fgets(line, sizeof line, maps)) mapped |= strstr(line, resolved) != NULL;
  if (maps) fclose(maps);
  char extra[64];
  snprintf(extra, sizeof extra, "\"dlclose_rc\":%d,\"mapped\":%s", rc, mapped ? "true" : "false");
  done("dlclose", NO_STATUS, NULL, extra);
  free(resolved);
  return 0;
}

// init, then exit normally: static destructors of the library run now.
int scenario_exit_without_cleanup(void) {
  do_init("init", opt.config, 1);
  begin("exiting");
  done("exiting", NO_STATUS, NULL, NULL);
  exit(0);
}

// Failing calls and what the runtime says about them (needs init, no model): errors after a later success, bad model
// files, calls after cleanup.
int scenario_errors(void) {
  ModelConfig preinit_model = {.file_path = "/nonexistent/oaax-conformance.model"};
  begin("preinit_load_models");  // leaves an error message behind: the next successful call must not keep it
  done("preinit_load_models", rt.load_models(1, &preinit_model), NULL, NULL);
  do_init("init", opt.config, 1);
  do_string("get_error_after_init", rt.get_error);

  ModelConfig missing = {.file_path = "/nonexistent/oaax-conformance.model"};
  begin("load_missing_file");
  done("load_missing_file", rt.load_models(1, &missing), NULL, NULL);

  char corrupt_path[] = "/tmp/oaax-corrupt-XXXXXX";
  int fd = mkstemp(corrupt_path);
  if (fd >= 0) {
    const char junk[] = "this is not a model: oaax conformance suite garbage bytes";
    if (write(fd, junk, sizeof junk) < 0) perror("write");
    close(fd);
  }
  ModelConfig corrupt = {.file_path = corrupt_path};
  begin("load_corrupt_file");
  done("load_corrupt_file", rt.load_models(1, &corrupt), NULL, NULL);
  unlink(corrupt_path);

  do_cleanup("cleanup");
  begin("post_cleanup_load_models");
  done("post_cleanup_load_models", rt.load_models(1, &missing), NULL, NULL);
  Tensors *input = make_input();
  begin("post_cleanup_enqueue_input");
  RuntimeStatus s = rt.enqueue_input(0, input);
  done("post_cleanup_enqueue_input", s, NULL, NULL);
  if (s != RUNTIME_STATUS_SUCCESS) free_tensors(input);
  int model_id = -1;
  Tensors *output = NULL;
  begin("post_cleanup_retrieve_output");
  done("post_cleanup_retrieve_output", rt.retrieve_output(&model_id, &output, 0), NULL, NULL);
  free_tensors(output);
  return 0;
}

// Its own scenario: NULL pointers are the likeliest way to crash a runtime, and a crash ends the scenario.
int scenario_null_pointers(void) {
  do_init("init", opt.config, 1);
  int model_id = -1;
  Tensors *output = NULL;
  begin("retrieve_null_model_id");
  done("retrieve_null_model_id", rt.retrieve_output(NULL, &output, 0), NULL, NULL);
  begin("retrieve_null_output");
  done("retrieve_null_output", rt.retrieve_output(&model_id, NULL, 0), NULL, NULL);
  do_cleanup("cleanup");
  return 0;
}

// Model tier. Loading the model the user passed with -m is the precondition of every scenario here: if it fails, the
// scenario ends with EXIT_MODEL and its checks are "blocked", not failed.
void load_user_model(const char *step) {
  ModelConfig model = {.file_path = opt.model};
  begin(step);
  RuntimeStatus s = rt.load_models(1, &model);
  done(step, s, NULL, NULL);
  if (s != RUNTIME_STATUS_SUCCESS) exit(EXIT_MODEL);
}

// A second load_models call after a successful one.
int scenario_load_twice(void) {
  do_init("init", opt.config, 1);
  load_user_model("load_first");
  ModelConfig model = {.file_path = opt.model};
  begin("load_second");
  done("load_second", rt.load_models(1, &model), NULL, NULL);
  do_cleanup("cleanup");
  return 0;
}

