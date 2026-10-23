// Shared by the harness sources: the runtime under test, the options, and the observation helpers.
#pragma once
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


#define NO_STATUS INT_MIN

struct Runtime {
  init_fn init;
  load_fn load_models;
  enqueue_fn enqueue_input;
  retrieve_fn retrieve_output;
  cleanup_fn cleanup;
  string_fn get_error, get_version, get_name, get_info;
};
extern struct Runtime rt;

// One model input as given with -i name:DTYPE:d1,d2,...  (dtype names of the spec; a dynamic dimension is passed as 1)
struct InputSpec {
  char name[256];
  int dtype;
  size_t elem;  // bytes per element
  int rank;
  int shape[8];
  int dynamic[8];  // 1 where the model accepts any size (sent as 1)
};

struct Options {
  const char *scenario, *lib, *out, *model, *dump;
  int cycles, settle_ms;
  int flood, requests, producers, per_producer, model_cycles, timeout_ms, lost_after_ms;  // load sizes: --hardware-safe shrinks them
  int budget_ms;  // loops stop early once the scenario has used this much time (0 = no limit): a slow device is not a failing one
  int warmup_ms;  // how long the very first request may take: the other timeouts are scaled from it
  struct InputSpec inputs[16];
  int num_inputs;
  Config config;  // from -c key=value
};
extern struct Options opt;
extern FILE *out;
extern void *lib_handle;

void put_hex(const char *s);
void put_hex_bytes(const char *s, size_t n);
void end_marker(const char *reason);  // the last line of every run: why the process is about to end (ok, dlopen, symbols, init, model)
double now_ms(void);
int out_of_time(double started_ms);   // true once the --budget-ms of the scenario is used up
void begin(const char *step);
void done(const char *step, int status, const char *str, const char *extra);
void thread_names(char *buf, size_t size);
long rss_bytes(void);
int thread_count(void);
void settle(int baseline, int ms);
int load_library(void);
Tensors *make_input(void);
void free_tensors(Tensors *t);
Config user_config_plus(const char *extra_key, const char *extra_value);
int do_init(const char *step, Config config, int fatal);
int do_cleanup(const char *step);
void do_string(const char *step, string_fn fn);
void load_user_model(const char *step);  // load -m MODEL or end the scenario with EXIT_MODEL
