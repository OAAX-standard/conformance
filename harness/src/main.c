// OAAX conformance harness: runs one scenario against a runtime library and writes what it observed
// (one JSON line per call) to --out. It makes no pass/fail decisions: those live in the Python checks.
// Each call is announced by a {"begin": ...} line before it, so a crash is attributed to the call that caused it.
#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "interface.h"

enum { EXIT_DLOPEN = 3, EXIT_SYMBOLS = 4, EXIT_INIT = 5, EXIT_MODEL = 6, EXIT_USAGE = 2 };
#define NO_STATUS INT_MIN

typedef RuntimeStatus (*init_fn)(Config);
typedef RuntimeStatus (*load_fn)(int, const ModelConfig *);
typedef RuntimeStatus (*enqueue_fn)(int, Tensors *);
typedef RuntimeStatus (*retrieve_fn)(int *, Tensors **, int);
typedef RuntimeStatus (*cleanup_fn)(void);
typedef const char *(*string_fn)(void);

static struct {
  init_fn init;
  load_fn load_models;
  enqueue_fn enqueue_input;
  retrieve_fn retrieve_output;
  cleanup_fn cleanup;
  string_fn get_error, get_version, get_name, get_info;
} rt;

static const struct { const char *name; void **slot; } SYMBOLS[] = {
    {"runtime_init", (void **)&rt.init},
    {"runtime_load_models", (void **)&rt.load_models},
    {"runtime_enqueue_input", (void **)&rt.enqueue_input},
    {"runtime_retrieve_output", (void **)&rt.retrieve_output},
    {"runtime_cleanup", (void **)&rt.cleanup},
    {"runtime_get_error", (void **)&rt.get_error},
    {"runtime_get_version", (void **)&rt.get_version},
    {"runtime_get_name", (void **)&rt.get_name},
    {"runtime_get_info", (void **)&rt.get_info},
};
#define NUM_SYMBOLS (sizeof SYMBOLS / sizeof SYMBOLS[0])

static struct {
  const char *scenario, *lib, *out, *model;
  int cycles, settle_ms;
  Config config;  // from -c key=value
} opt = {.cycles = 50, .settle_ms = 3000};

static FILE *out;
static void *lib_handle;

// ---- observations -------------------------------------------------------------------------------------------

// Runtime strings are bytes of unknown encoding (UTF-8 validity is itself checked), so they travel as hex.
static void put_hex(const char *s) {
  if (!s) {
    fputs("null", out);
    return;
  }
  fputc('"', out);
  for (; *s; s++) fprintf(out, "%02x", (unsigned char)*s);
  fputc('"', out);
}

static void begin(const char *step) {
  fprintf(out, "{\"begin\":\"%s\"}\n", step);
  fflush(out);
}

// {"step": name, "status": n|null, "str": hex|null [, "err": hex] [, extra members]}
static void done(const char *step, int status, const char *str, const char *extra) {
  fprintf(out, "{\"step\":\"%s\",\"status\":", step);
  if (status == NO_STATUS) fputs("null", out);
  else fprintf(out, "%d", status);
  fputs(",\"str\":", out);
  put_hex(str);
  if (status != NO_STATUS && status != RUNTIME_STATUS_SUCCESS) {
    fputs(",\"err\":", out);
    put_hex(rt.get_error());
  }
  if (extra) fprintf(out, ",%s", extra);
  fputs("}\n", out);
  fflush(out);
}

// Thread names (/proc/self/task/*/comm) as a JSON array.
static void thread_names(char *buf, size_t size) {
  size_t n = 0;
  n += snprintf(buf + n, size - n, "[");
  DIR *dir = opendir("/proc/self/task");
  struct dirent *e;
  int first = 1;
  while (dir && (e = readdir(dir)) && n + 80 < size) {
    if (e->d_name[0] == '.') continue;
    char path[320], name[64] = "?";
    snprintf(path, sizeof path, "/proc/self/task/%s/comm", e->d_name);
    FILE *f = fopen(path, "r");
    if (f) {
      if (fgets(name, sizeof name, f)) name[strcspn(name, "\n\"\\")] = 0;
      fclose(f);
    }
    n += snprintf(buf + n, size - n, "%s\"%.40s\"", first ? "" : ",", name);
    first = 0;
  }
  if (dir) closedir(dir);
  snprintf(buf + n, size - n, "]");
}

static long rss_bytes(void) {
  long pages = 0, resident = 0;
  FILE *f = fopen("/proc/self/statm", "r");
  if (!f) return -1;
  if (fscanf(f, "%ld %ld", &pages, &resident) != 2) resident = -1;
  fclose(f);
  return resident < 0 ? -1 : resident * sysconf(_SC_PAGESIZE);
}

static int thread_count(void) {
  char buf[8192];
  thread_names(buf, sizeof buf);
  int n = 0;
  for (char *p = buf; *p; p++) n += *p == '"';
  return n / 2;
}

// Poll until the process is back to `baseline` threads or `ms` have passed.
static void settle(int baseline, int ms) {
  for (int waited = 0; waited < ms && thread_count() > baseline; waited += 100) usleep(100 * 1000);
}

// ---- loading ------------------------------------------------------------------------------------------------

static int load_library(void) {
  begin("dlopen");
  lib_handle = dlopen(opt.lib, RTLD_NOW | RTLD_LOCAL);
  if (!lib_handle) {
    done("dlopen", NO_STATUS, dlerror(), "\"ok\":false");
    return EXIT_DLOPEN;
  }
  done("dlopen", NO_STATUS, NULL, "\"ok\":true");

  char present[1024] = "", missing[1024] = "";
  for (size_t i = 0; i < NUM_SYMBOLS; i++) {
    *SYMBOLS[i].slot = dlsym(lib_handle, SYMBOLS[i].name);
    char *list = *SYMBOLS[i].slot ? present : missing;
    snprintf(list + strlen(list), 1024 - strlen(list), "%s\"%s\"", *list ? "," : "", SYMBOLS[i].name);
  }
  char extra[2100];
  snprintf(extra, sizeof extra, "\"present\":[%s],\"missing\":[%s]", present, missing);
  done("symbols", NO_STATUS, NULL, extra);
  return *missing ? EXIT_SYMBOLS : 0;
}

// ---- tensors ------------------------------------------------------------------------------------------------

static Tensors *make_input(void) {
  Tensors *t = calloc(1, sizeof *t);
  t->id = 1;
  t->num_tensors = 1;
  t->tensors = calloc(1, sizeof *t->tensors);
  t->tensors[0].name = strdup("input");
  t->tensors[0].data_type = DATA_TYPE_FLOAT;
  t->tensors[0].rank = 1;
  t->tensors[0].shape = malloc(sizeof(int));
  t->tensors[0].shape[0] = 1;
  t->tensors[0].data_size = sizeof(float);
  t->tensors[0].data = calloc(1, sizeof(float));
  return t;
}

static void free_tensors(Tensors *t) {
  if (!t) return;
  for (int i = 0; i < t->num_tensors; i++) {
    free(t->tensors[i].name);
    free(t->tensors[i].shape);
    free(t->tensors[i].data);
  }
  free(t->tensors);
  free(t);
}

// ---- configuration ------------------------------------------------------------------------------------------

static Config user_config_plus(const char *extra_key, const char *extra_value) {
  Config c = opt.config;
  if (!extra_key) return c;
  c.keys = malloc((c.length + 1) * sizeof *c.keys);
  c.values = malloc((c.length + 1) * sizeof *c.values);
  memcpy(c.keys, opt.config.keys, c.length * sizeof *c.keys);
  memcpy(c.values, opt.config.values, c.length * sizeof *c.values);
  c.keys[c.length] = extra_key;
  c.values[c.length] = extra_value;
  c.length++;
  return c;
}

// One init call with `config`; records it as `step` and exits if `fatal` and it failed.
static int do_init(const char *step, Config config, int fatal) {
  begin(step);
  RuntimeStatus s = rt.init(config);
  done(step, s, NULL, NULL);
  if (fatal && s != RUNTIME_STATUS_SUCCESS) exit(EXIT_INIT);
  return s;
}

static int do_cleanup(const char *step) {
  begin(step);
  RuntimeStatus s = rt.cleanup();
  done(step, s, NULL, NULL);
  return s;
}

static void do_string(const char *step, string_fn fn) {
  begin(step);
  done(step, NO_STATUS, fn(), "\"getter\":true");  // lets reports show NULL instead of an empty cell
}

// ---- scenarios ----------------------------------------------------------------------------------------------

// No init: getters, info/error before init, cleanup, calls before init.
static int scenario_static(void) {
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
static int scenario_lifecycle(void) {
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
static int scenario_cycles(void) {
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
static int scenario_unload(void) {
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
static int scenario_exit_without_cleanup(void) {
  do_init("init", opt.config, 1);
  begin("exiting");
  done("exiting", NO_STATUS, NULL, NULL);
  exit(0);
}

// Failing calls and what the runtime says about them (needs init, no model): errors after a later success, bad model
// files, calls after cleanup.
static int scenario_errors(void) {
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
static int scenario_null_pointers(void) {
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
static void load_user_model(const char *step) {
  ModelConfig model = {.file_path = opt.model};
  begin(step);
  RuntimeStatus s = rt.load_models(1, &model);
  done(step, s, NULL, NULL);
  if (s != RUNTIME_STATUS_SUCCESS) exit(EXIT_MODEL);
}

// A second load_models call after a successful one.
static int scenario_load_twice(void) {
  do_init("init", opt.config, 1);
  load_user_model("load_first");
  ModelConfig model = {.file_path = opt.model};
  begin("load_second");
  done("load_second", rt.load_models(1, &model), NULL, NULL);
  do_cleanup("cleanup");
  return 0;
}

static const struct { const char *name; int (*run)(void); } SCENARIOS[] = {
    {"static", scenario_static},
    {"lifecycle", scenario_lifecycle},
    {"cycles", scenario_cycles},
    {"unload", scenario_unload},
    {"exit-without-cleanup", scenario_exit_without_cleanup},
    {"errors", scenario_errors},
    {"null-pointers", scenario_null_pointers},
    {"load-twice", scenario_load_twice},
};

// ---- main ---------------------------------------------------------------------------------------------------

static int usage(void) {
  fprintf(stderr, "usage: oaax-harness <scenario> <library> --out FILE [-c key=value]... [-m MODEL] [--cycles N] [--settle-ms N]\n");
  return EXIT_USAGE;
}

static int parse(int argc, char **argv) {
  if (argc < 3) return usage();
  opt.scenario = argv[1];
  opt.lib = argv[2];
  opt.config.keys = calloc(argc, sizeof(char *));
  opt.config.values = calloc(argc, sizeof(char *));
  for (int i = 3; i < argc; i++) {
    int has_value = i + 1 < argc;
    if (!strcmp(argv[i], "--out") && has_value) opt.out = argv[++i];
    else if (!strcmp(argv[i], "--cycles") && has_value) opt.cycles = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--settle-ms") && has_value) opt.settle_ms = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-m") && has_value) opt.model = argv[++i];
    else if (!strcmp(argv[i], "-c") && has_value) {
      char *eq = strchr(argv[++i], '=');
      if (!eq) return usage();
      *eq = 0;
      opt.config.keys[opt.config.length] = argv[i];
      opt.config.values[opt.config.length++] = eq + 1;
    } else return usage();
  }
  return opt.out ? 0 : usage();
}

int main(int argc, char **argv) {
  int rc = parse(argc, argv);
  if (rc) return rc;
  out = fopen(opt.out, "w");
  if (!out) {
    perror(opt.out);
    return EXIT_USAGE;
  }
  rc = load_library();
  if (rc) return rc;
  for (size_t i = 0; i < sizeof SCENARIOS / sizeof SCENARIOS[0]; i++)
    if (!strcmp(opt.scenario, SCENARIOS[i].name)) return SCENARIOS[i].run();
  fprintf(stderr, "unknown scenario: %s\n", opt.scenario);
  return EXIT_USAGE;
}
