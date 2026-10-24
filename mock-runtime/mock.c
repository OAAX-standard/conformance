// A tiny OAAX 2.0 runtime for testing the conformance suite itself. It is correct unless the environment variable
// OAAX_MOCK_FAULT names a fault: every check of the suite must fail on its fault and pass on the clean mock.
//
// A "model" is a file that starts with MOCKMODEL. It takes two inputs, "x" FLOAT [1,4] and "scale" FLOAT [1], and returns
// "y" FLOAT [1,4] (= x * scale) and "sum" FLOAT [1]; a request takes 5 ms. Build variants (CMakeLists.txt) cover missing and
// extra exports. The faults are listed next to the code they change.
#include <signal.h>
#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "interface.h"

#define EXPORT __attribute__((visibility("default")))
#define MAX_MODELS 8
#define INFERENCE_MS 5

static int initialized, cleaned_once, has_error, models_loaded, num_models, load_broken, generation, accepted_total;
static char error_text[480];
static char config_value[160];  // fault echo_config: the runtime repeats the configuration it was given in its error text
static int first_request_done;
static const unsigned char *kept_data[MAX_MODELS];  // fault retain_buffer: borrowed instead of copied
static const char *kept_path[MAX_MODELS];

struct Request {
  int id, model_id;
  double ready_ms;
  float y[4], sum;
  struct Request *next;
};
static struct Request *head, *tail;
static int queued;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t changed = PTHREAD_COND_INITIALIZER;

static int fault(const char *name) {
  const char *f = getenv("OAAX_MOCK_FAULT");
  return f && !strcmp(f, name);
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// A real leak: the volatile sink stops the compiler from deleting an allocation nobody uses.
static void *volatile leak_sink;
static void leak(size_t bytes) {
  char *p = malloc(bytes);
  memset(p, 1, bytes);
  leak_sink = p;
}

static void sleep_ms(int ms) {
  struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
  nanosleep(&ts, NULL);
}

static RuntimeStatus fail(RuntimeStatus status, const char *message) {
  if (!fault("no_error_text")) {
    snprintf(error_text, sizeof error_text, "%s%s%s%s", fault("ctrl_err") ? "\x01\x1b[31m" : "", message, fault("echo_config") ? " config=" : "",
             fault("echo_config") ? config_value : "");
    has_error = 1;
  }
  return status;
}

static void *idle_thread(void *arg) {
  (void)arg;
  pthread_setname_np(pthread_self(), fault("bad_thread_name") ? "\xff" "\x01" "bad" : "mock-worker");
  for (;;) pause();
  return NULL;
}

// A call while not initialized (before init, or after cleanup): NOT_INITIALIZED, unless a fault says otherwise.
static RuntimeStatus not_initialized(void) {
  if (fault("preinit_success") || (cleaned_once && fault("postcleanup_success"))) return RUNTIME_STATUS_SUCCESS;
  if (fault("crash_preinit")) *(volatile int *)0 = 1;
  return fail(RUNTIME_STATUS_NOT_INITIALIZED, "runtime not initialized");
}

// ---- lifecycle -------------------------------------------------------------------------------------------

EXPORT RuntimeStatus runtime_init(Config config) {
  if (initialized) {
    if (fault("init_twice_ok")) return RUNTIME_STATUS_SUCCESS;
    return fail(RUNTIME_STATUS_ALREADY_INITIALIZED, "already initialized");
  }
  int has_key = 0;
  for (int i = 0; i < config.length; i++) {
    has_key |= !strcmp(config.keys[i], "vendor_key");  // fault require_config: the runtime cannot start without this vendor key
    if (fault("echo_config")) snprintf(config_value, sizeof config_value, "%s", config.values[i]);
  }
  if (fault("require_config") && !has_key) return fail(RUNTIME_STATUS_ERROR, "config required");
  if (fault("slow_init")) sleep_ms(2500);
  if (fault("hang_init")) sleep_ms(600000);
  if (fault("chatty")) {
    for (int i = 0; i < 400; i++) fprintf(stderr, "mock runtime: noisy line %03d ......................................................\n", i);
    fflush(stderr);
  }
  if (fault("daemonize") && fork() == 0) {  // a helper process that keeps the inherited stdout/stderr open
    sleep(300);
    _exit(0);
  }
  for (int i = 0; i < config.length; i++)
    if (fault("reject_unknown_key") && !strcmp(config.keys[i], "oaax_conformance_unknown_key"))
      return fail(RUNTIME_STATUS_INVALID_ARGUMENT, "unknown key");
  if (cleaned_once && fault("no_reinit")) return fail(RUNTIME_STATUS_ERROR, "cannot re-initialize");

  if (fault("leak_thread") || fault("bad_thread_name")) {
    pthread_t t;
    pthread_create(&t, NULL, idle_thread, NULL);
    pthread_detach(t);
  }
  if (fault("leak_memory")) leak(2 << 20);
  if (fault("pin_library")) {
    Dl_info info;
    if (dladdr((void *)runtime_init, &info)) dlopen(info.dli_fname, RTLD_NOW | RTLD_NODELETE);
  }
  initialized = 1;
  load_broken = 0;
  first_request_done = 0;
  if (!fault("stale_error")) has_error = 0;  // a successful call leaves no error behind
  return RUNTIME_STATUS_SUCCESS;
}

static void drop_queue(void) {
  while (head) {
    struct Request *next = head->next;
    free(head);
    head = next;
  }
  tail = NULL;
  queued = 0;
}

EXPORT RuntimeStatus runtime_cleanup(void) {
  if (fault("crash_in_cleanup")) raise(SIGSEGV);
  if (!initialized && fault("cleanup_not_idempotent")) return fail(RUNTIME_STATUS_ERROR, "not initialized");
  if (!initialized && cleaned_once && fault("cleanup_twice_fails")) return fail(RUNTIME_STATUS_ERROR, "already cleaned up");
  pthread_mutex_lock(&lock);
  if (queued && fault("cleanup_fails_inflight")) {
    pthread_mutex_unlock(&lock);
    return fail(RUNTIME_STATUS_ERROR, "requests in flight");
  }
  initialized = 0;
  models_loaded = 0;
  num_models = 0;
  cleaned_once = 1;
  generation++;
  drop_queue();
  if (!fault("no_wakeup")) pthread_cond_broadcast(&changed);  // a blocked retrieve_output must return
  pthread_mutex_unlock(&lock);
  return RUNTIME_STATUS_SUCCESS;
}

// ---- models ----------------------------------------------------------------------------------------------

static int valid_model(const ModelConfig *m) {
  char magic[9] = "";
  if (m->model_data) return m->model_size >= 9 && !memcmp(m->model_data, "MOCKMODEL", 9);
  FILE *f = fopen(m->file_path, "rb");
  if (!f) return 0;
  size_t n = fread(magic, 1, sizeof magic, f);
  fclose(f);
  return n == sizeof magic && !memcmp(magic, "MOCKMODEL", sizeof magic);
}

// Loading is once per init: a later call is an error. A missing file is FILE_NOT_FOUND, any other file INVALID_MODEL.
EXPORT RuntimeStatus runtime_load_models(int count, const ModelConfig *configs) {
  if (!initialized) return not_initialized();
  if (count < 1 || count > MAX_MODELS || !configs) return fail(RUNTIME_STATUS_INVALID_ARGUMENT, "no model given");
  if (fault("exit_in_call")) exit(3);  // a runtime that ends the process from inside a call
  if (fault("one_model") && count > 1) return fail(RUNTIME_STATUS_OUT_OF_MEMORY, "the device holds one model at a time");
  if (models_loaded && !fault("load_twice_ok")) return fail(RUNTIME_STATUS_ALREADY_INITIALIZED, "models are already loaded");
  if (load_broken) return fail(RUNTIME_STATUS_ERROR, "a failed load left the runtime unusable");  // fault load_sticky_failure
  for (int i = 0; i < count; i++) {
    if (!configs[i].file_path && !configs[i].model_data) return fail(RUNTIME_STATUS_INVALID_ARGUMENT, "no model given");
    if (configs[i].model_data && fault("no_memory_load")) return fail(RUNTIME_STATUS_NOT_IMPLEMENTED, "model_data is not supported");
    if (fault("load_error_generic")) return fail(RUNTIME_STATUS_ERROR, "failed to load the model");
    if (!configs[i].model_data && access(configs[i].file_path, R_OK)) {
      if (fault("load_sticky_failure")) load_broken = 1;
      if (fault("weird_status")) return fail((RuntimeStatus)99, "a status the header does not define");
      return fail(RUNTIME_STATUS_FILE_NOT_FOUND, "model file not found");
    }
    if (!valid_model(&configs[i])) return fail(RUNTIME_STATUS_INVALID_MODEL, "not a model");
  }
  for (int i = 0; i < count; i++) {
    kept_data[i] = configs[i].model_data;  // used only by the retain_buffer fault
    kept_path[i] = configs[i].file_path;
  }
  if (fault("leak_per_load")) leak(4 << 20);
  models_loaded = 1;
  num_models = count;
  return RUNTIME_STATUS_SUCCESS;
}

// ---- inference -------------------------------------------------------------------------------------------

static const TensorDescriptor *find(const Tensors *t, const char *name) {
  for (int i = 0; i < t->num_tensors; i++)
    if (t->tensors[i].name && !strcmp(t->tensors[i].name, name)) return &t->tensors[i];
  return NULL;
}

// NULL if the request is fine, else the status and message. Faults make the mock accept what it should reject.
static RuntimeStatus validate(const Tensors *t, const char **why) {
  if (!t) return (*why = "null input_tensors"), RUNTIME_STATUS_INVALID_ARGUMENT;
  if (t->num_tensors == 0) {
    if (fault("accept_empty")) return RUNTIME_STATUS_SUCCESS;
    return (*why = "no tensors"), RUNTIME_STATUS_INVALID_TENSOR;
  }
  const TensorDescriptor *x = find(t, "x"), *scale = find(t, "scale");
  if (!x || !scale) {
    if (t->num_tensors == 1 ? fault("accept_missing") : fault("accept_unknown_name")) return RUNTIME_STATUS_SUCCESS;
    return (*why = "missing or unknown input"), RUNTIME_STATUS_INVALID_TENSOR;
  }
  if (fault("u8_only")) {  // takes uint8 pixels where the ONNX model has floats
    if (x->data_type != DATA_TYPE_UINT8) return (*why = "x must be UINT8"), RUNTIME_STATUS_TENSOR_TYPE_MISMATCH;
    return RUNTIME_STATUS_SUCCESS;
  }
  if (fault("any_shape")) return RUNTIME_STATUS_SUCCESS;  // every dimension dynamic
  if (x->data_type != DATA_TYPE_FLOAT || scale->data_type != DATA_TYPE_FLOAT) {
    if (fault("accept_wrong_dtype")) return RUNTIME_STATUS_SUCCESS;
    return (*why = "wrong data type"), fault("type_code_generic") ? RUNTIME_STATUS_ERROR : RUNTIME_STATUS_TENSOR_TYPE_MISMATCH;
  }
  int dim0_ok = fault("dynamic_dim0") ? x->shape[0] >= 1 : x->shape[0] == 1;  // fault dynamic_dim0: the first dimension takes any size
  if (x->rank != 2 || !dim0_ok || x->shape[1] != 4 || scale->rank != 1 || scale->shape[0] != 1) {
    if (fault("accept_wrong_shape")) return RUNTIME_STATUS_SUCCESS;
    return (*why = "wrong shape"), fault("shape_code_generic") ? RUNTIME_STATUS_ERROR : RUNTIME_STATUS_TENSOR_SHAPE_MISMATCH;
  }
  if (x->data_size != (size_t)(x->shape[0] * 16) || scale->data_size != 4) {
    if (fault("accept_bad_size")) return RUNTIME_STATUS_SUCCESS;
    return (*why = "data_size does not match shape and type"), RUNTIME_STATUS_INVALID_TENSOR;
  }
  return RUNTIME_STATUS_SUCCESS;
}

static int still_valid_model(int model_id) {  // fault retain_buffer: the runtime reads the buffers it should have copied
  if (!fault("retain_buffer")) return 1;
  if (kept_data[model_id]) return !memcmp(kept_data[model_id], "MOCKMODEL", 9);
  return kept_path[model_id] && access(kept_path[model_id], R_OK) == 0;
}

static void free_input(Tensors *t) {
  for (int i = 0; i < t->num_tensors; i++) free(t->tensors[i].name), free(t->tensors[i].shape), free(t->tensors[i].data);
  free(t->tensors);
  free(t);
}

// Does the request really match the model? (a fault can let a request through that does not)
static int usable(const Tensors *t) {
  const TensorDescriptor *x = t ? find(t, "x") : NULL, *scale = t ? find(t, "scale") : NULL;
  if (!x || !scale) return 0;
  if (fault("u8_only") || fault("any_shape")) return x->rank == 2 && x->shape[1] == 4 && x->data_size >= 4;
  return x->data_type == DATA_TYPE_FLOAT && x->rank == 2 && x->shape[1] == 4 && x->data_size == (size_t)(x->shape[0] * 16) && scale->data_size == 4;
}

EXPORT RuntimeStatus runtime_enqueue_input(int model_id, Tensors *input) {
  if (!initialized) return not_initialized();
  if (fault("null_tensors_crash")) (void)*(volatile int *)&input->id;
  if (!models_loaded) return fail(RUNTIME_STATUS_MODEL_NOT_LOADED, "no model loaded");
  if (fault("enqueue_blocks")) sleep_ms(20);  // the caller waits for the inference (and more)
  if (model_id < 0 || model_id >= num_models) {
    if (!fault("accept_bad_id")) return fail(fault("bad_id_generic") ? RUNTIME_STATUS_ERROR : RUNTIME_STATUS_INVALID_MODEL_ID, "invalid model_id");
    model_id = 0;
  }
  const char *why = "";
  RuntimeStatus bad = validate(input, &why);
  if (bad != RUNTIME_STATUS_SUCCESS) {
    if (fault("break_after_reject")) load_broken = 1;
    if (input && fault("free_rejected")) free_input(input);  // fault: the runtime frees an input it rejected
    return fail(bad, why);
  }
  if (load_broken && fault("break_after_reject")) return fail(RUNTIME_STATUS_ERROR, "broken by an earlier rejection");
  if (!still_valid_model(model_id)) return fail(RUNTIME_STATUS_ERROR, "the model buffer was not copied");
  if (fault("leak_per_request")) leak(256 * 1024);

  struct Request *r = NULL;  // fault accept_*: a request that does not match is accepted and then silently lost
  if (usable(input)) {
    r = calloc(1, sizeof *r);
    r->id = input->id;
    r->model_id = fault("wrong_model_id") ? 0 : model_id;
    const TensorDescriptor *xd = find(input, "x");
    float scale = fault("u8_only") || fault("any_shape") ? 1.0f : *(const float *)find(input, "scale")->data;
    for (int i = 0; i < 4; i++) {
      float xi = xd->data_type == DATA_TYPE_UINT8 ? ((const unsigned char *)xd->data)[i] : ((const float *)xd->data)[i];
      r->y[i] = xi * scale;
      r->sum += r->y[i];
    }
    if (fault("jitter")) r->y[1] += 1e-7f * (float)(input->id % 5);  // below the comparison tolerance
    static int corrupt_count;
    if (fault("corrupt_every_7th") && ++corrupt_count % 7 == 0) r->y[0] += 1.0f;
    if (fault("wrong_output")) r->y[0] += 1.0f;
    r->ready_ms = now_ms() + (fault("slow_request") ? 150 : INFERENCE_MS) + (fault("slow_first") && !first_request_done ? 3000 : 0);
    first_request_done = 1;
  }
  pthread_mutex_lock(&lock);
  accepted_total++;
  int drop = (fault("concurrency_drop") && queued >= 2 && accepted_total % 3 == 0)  /* only under a backlog */ || (fault("flood_lost") && queued > 50);
  if (r && !drop && !fault("drop_all")) {
    if (tail) tail->next = r;
    else head = r;
    tail = r;
    queued++;
    pthread_cond_broadcast(&changed);
  } else free(r);
  pthread_mutex_unlock(&lock);
  free_input(input);  // success: the runtime owns the input
  return RUNTIME_STATUS_SUCCESS;
}

static void fill_descriptor(TensorDescriptor *d, const char *name, int rank, int n, const float *values) {
  if (fault("huge_name") && !strcmp(name, "y")) {  // a 40000-byte tensor name
    d->name = malloc(40001);
    memset(d->name, 'n', 40000);
    d->name[40000] = 0;
  } else d->name = fault("output_null_name") && !strcmp(name, "y") ? NULL : strdup(fault("rename_output") ? "out" : name);
  d->data_type = DATA_TYPE_FLOAT;
  d->rank = rank;
  d->shape = malloc(rank * sizeof(int));
  d->shape[0] = 1;
  if (rank == 2) d->shape[1] = n;
  d->data_size = n * sizeof(float);
  d->data = malloc(d->data_size);
  memcpy(d->data, values, d->data_size);
}

static Tensors *make_output(const struct Request *r) {
  Tensors *t = calloc(1, sizeof *t);
  t->id = fault("no_id_echo") ? 0 : r->id;
  t->num_tensors = 2;
  t->tensors = calloc(2, sizeof *t->tensors);
  fill_descriptor(&t->tensors[0], "y", 2, 4, r->y);
  fill_descriptor(&t->tensors[1], "sum", 2, 1, &r->sum);  // shape [1,1]
  if (fault("output_aliased")) {
    free(t->tensors[1].data);
    t->tensors[1].data = t->tensors[0].data;  // two descriptors, one buffer: the host would free it twice
  }
  if (fault("bad_data_size")) t->tensors[0].data_size = 8;
  if (fault("big_size")) t->tensors[0].data_size = 1600000;  // claims far more than it allocated
  return t;
}

EXPORT RuntimeStatus runtime_retrieve_output(int *model_id, Tensors **output_tensors, int timeout_ms) {
  if (fault("null_ptr_crash")) *output_tensors = NULL;  // dereferences the pointer the caller left NULL
  if (!model_id || !output_tensors) {
    if (fault("null_ptr_ignored")) return RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
    return fail(RUNTIME_STATUS_INVALID_ARGUMENT, "model_id and output_tensors must not be NULL");
  }
  if (!initialized) return not_initialized();
  if (timeout_ms == 0 && fault("retrieve0_blocks")) sleep_ms(300);
  double started = now_ms(), deadline = timeout_ms > 0 ? started + timeout_ms : 0;
  int gen = generation;
  pthread_mutex_lock(&lock);
  for (;;) {
    double now = now_ms();
    if (!fault("no_wakeup") && gen != generation) {
      pthread_mutex_unlock(&lock);
      return fail(RUNTIME_STATUS_NOT_INITIALIZED, "cleaned up while waiting");
    }
    if (head && head->ready_ms <= now) break;
    if (timeout_ms == 0 || (timeout_ms < 0 && fault("negative_no_wait")) || (timeout_ms > 0 && fault("retrieve_returns_early") && !head)) {
      pthread_mutex_unlock(&lock);
      return RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
    }
    double limit = timeout_ms > 0 ? deadline + (fault("retrieve_timeout_long") ? 3000 : 0) : 0;
    if (timeout_ms > 0 && now >= limit) {
      pthread_mutex_unlock(&lock);
      return RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
    }
    double wake = head ? head->ready_ms : limit;  // wake when the head request is ready, or at the deadline
    if (head && limit && limit < wake) wake = limit;
    if (wake) {
      struct timespec ts;
      clock_gettime(CLOCK_REALTIME, &ts);
      double delta = wake - now;
      ts.tv_sec += (time_t)(delta / 1000);
      ts.tv_nsec += (long)((delta - (long)(delta / 1000) * 1000) * 1e6);
      if (ts.tv_nsec >= 1000000000L) ts.tv_sec++, ts.tv_nsec -= 1000000000L;
      pthread_cond_timedwait(&changed, &lock, &ts);
    } else pthread_cond_wait(&changed, &lock);
  }
  struct Request *r = head;
  head = r->next;
  if (!head) tail = NULL;
  queued--;
  pthread_mutex_unlock(&lock);
  *model_id = r->model_id;
  *output_tensors = make_output(r);
  if (fault("null_tensors")) (*output_tensors)->tensors = NULL;  // a count with no array
  if (fault("negative_tensors")) (*output_tensors)->num_tensors = -3;
  free(r);
  return RUNTIME_STATUS_SUCCESS;
}

// ---- getters ---------------------------------------------------------------------------------------------

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
  static __thread char info[128];
  if (initialized) {
    if (fault("info_not_json")) return "not json";
    pthread_mutex_lock(&lock);
    snprintf(info, sizeof info, "{\"loaded_models\":%d,\"requests_in_flight\":%d}", fault("info_stale") ? 0 : num_models,
             fault("info_stale") ? 0 : queued);
    pthread_mutex_unlock(&lock);
    return info;
  }
  if (fault("info_before_init") || (cleaned_once && fault("info_after_cleanup"))) return "{}";
  return NULL;
}
#endif

#ifdef MOCK_EXTRA_EXPORT
EXPORT int mock_extra_symbol(void) { return 1; }
#endif

__attribute__((destructor)) static void at_exit(void) {
  if ((initialized && fault("crash_at_exit")) || fault("crash_after_end")) abort();
}
