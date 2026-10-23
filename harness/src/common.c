// OAAX conformance harness: runs one scenario against a runtime library and writes what it observed
// (one JSON line per call) to --out. It makes no pass/fail decisions: those live in the Python checks.
// Each call is announced by a {"begin": ...} line before it, so a crash is attributed to the call that caused it.
#include "common.h"

struct Runtime rt;
struct Options opt = {.cycles = 50, .settle_ms = 3000, .flood = 1000, .requests = 500, .producers = 4, .per_producer = 25, .model_cycles = 20, .timeout_ms = 10000, .lost_after_ms = 3000};
FILE *out;
void *lib_handle;

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


// ---- observations -------------------------------------------------------------------------------------------

// Runtime strings are bytes of unknown encoding (UTF-8 validity is itself checked), so they travel as hex.
void put_hex(const char *s) {
  if (!s) {
    fputs("null", out);
    return;
  }
  fputc('"', out);
  for (; *s; s++) fprintf(out, "%02x", (unsigned char)*s);
  fputc('"', out);
}

void begin(const char *step) {
  fprintf(out, "{\"begin\":\"%s\"}\n", step);
  fflush(out);
}

// {"step": name, "status": n|null, "str": hex|null [, "err": hex] [, extra members]}
void done(const char *step, int status, const char *str, const char *extra) {
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
void thread_names(char *buf, size_t size) {
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

long rss_bytes(void) {
  long pages = 0, resident = 0;
  FILE *f = fopen("/proc/self/statm", "r");
  if (!f) return -1;
  if (fscanf(f, "%ld %ld", &pages, &resident) != 2) resident = -1;
  fclose(f);
  return resident < 0 ? -1 : resident * sysconf(_SC_PAGESIZE);
}

int thread_count(void) {
  char buf[8192];
  thread_names(buf, sizeof buf);
  int n = 0;
  for (char *p = buf; *p; p++) n += *p == '"';
  return n / 2;
}

// Poll until the process is back to `baseline` threads or `ms` have passed.
void settle(int baseline, int ms) {
  for (int waited = 0; waited < ms && thread_count() > baseline; waited += 100) usleep(100 * 1000);
}

// ---- loading ------------------------------------------------------------------------------------------------

int load_library(void) {
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

Tensors *make_input(void) {
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

void free_tensors(Tensors *t) {
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

Config user_config_plus(const char *extra_key, const char *extra_value) {
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
int do_init(const char *step, Config config, int fatal) {
  begin(step);
  RuntimeStatus s = rt.init(config);
  done(step, s, NULL, NULL);
  if (fatal && s != RUNTIME_STATUS_SUCCESS) exit(EXIT_INIT);
  return s;
}

int do_cleanup(const char *step) {
  begin(step);
  RuntimeStatus s = rt.cleanup();
  done(step, s, NULL, NULL);
  return s;
}

void do_string(const char *step, string_fn fn) {
  begin(step);
  done(step, NO_STATUS, fn(), "\"getter\":true");  // lets reports show NULL instead of an empty cell
}

