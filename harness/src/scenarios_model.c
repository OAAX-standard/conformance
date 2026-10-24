// Model tier scenarios: they send requests, so they need -m MODEL and -i for every input of the model.
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <time.h>

#include "common.h"
#include "scenarios.h"

// ---- inputs and outputs ----------------------------------------------------------------------------------

static const struct { const char *name; int type; size_t size; } DTYPES[] = {
    {"FLOAT", 1, 4},  {"UINT8", 2, 1},   {"INT8", 3, 1},   {"UINT16", 4, 2}, {"INT16", 5, 2},
    {"INT32", 6, 4},  {"INT64", 7, 8},   {"BOOL", 9, 1},   {"FLOAT16", 10, 2}, {"DOUBLE", 11, 8},
    {"UINT32", 12, 4}, {"UINT64", 13, 8}, {"BFLOAT16", 16, 2},
};

static size_t element_size(int dtype) {
  for (size_t i = 0; i < sizeof DTYPES / sizeof DTYPES[0]; i++)
    if (DTYPES[i].type == dtype) return DTYPES[i].size;
  return 0;
}

// "name:DTYPE:1,3,240,320": the name may contain ':' itself, so read from the right.
int parse_input(const char *arg, struct InputSpec *spec) {
  const char *dims = strrchr(arg, ':');
  if (!dims || dims == arg) return 0;
  const char *dtype = dims - 1;
  while (dtype > arg && *dtype != ':') dtype--;
  if (dtype == arg || (size_t)(dtype - arg) >= sizeof spec->name) return 0;
  memcpy(spec->name, arg, dtype - arg);
  spec->name[dtype - arg] = 0;
  char dtype_name[16] = "";
  snprintf(dtype_name, sizeof dtype_name, "%.*s", (int)(dims - dtype - 1), dtype + 1);
  for (size_t i = 0; i < sizeof DTYPES / sizeof DTYPES[0]; i++)
    if (!strcmp(DTYPES[i].name, dtype_name)) spec->dtype = DTYPES[i].type, spec->elem = DTYPES[i].size;
  if (!spec->elem) return 0;
  spec->rank = 0;
  for (const char *p = dims + 1; *p && spec->rank < 8;) {
    spec->dynamic[spec->rank] = *p == '?';
    spec->shape[spec->rank++] = *p == '?' ? 1 : atoi(p);
    p = strchr(p, ',');
    if (!p) break;
    p++;
  }
  return 1;
}

static size_t element_count(const int *shape, int rank) {
  size_t n = 1;
  for (int i = 0; i < rank; i++)
    if (shape[i] < 0 || __builtin_mul_overflow(n, (size_t)shape[i], &n)) return 0;  // a runtime's shape may be anything: 0 = not countable
  return n;
}

// Deterministic data: floats in [0,1), small integers, false for BOOL.
static void fill(void *data, size_t bytes, int dtype) {
  uint32_t state = 12345;
  unsigned char *p = data;
  size_t elem = element_size(dtype);
  for (size_t off = 0; elem && off + elem <= bytes; off += elem) {
    state = state * 1664525u + 1013904223u;
    float f = (float)(state >> 8) / 16777216.0f;
    if (dtype == 1) memcpy(p + off, &f, 4);
    else if (dtype == 11) { double d = f; memcpy(p + off, &d, 8); }
    else if (dtype == 10) { uint16_t h = 0x3800; memcpy(p + off, &h, 2); }
    else if (dtype == 16) { uint16_t h = 0x3F00; memcpy(p + off, &h, 2); }
    else if (dtype == 9) memset(p + off, 0, elem);
    else { memset(p + off, 0, elem); p[off] = (unsigned char)((state >> 16) % 7); }
  }
}

// How a request deviates from what the model takes (all zero = valid).
struct Build {
  int empty;              // no tensors at all
  int drop_last;          // leave out the last input
  const char *first_name; // rename the first input
  int other_dtype;        // give the first input another data type
  int other_shape;        // make one fixed dimension of one input one larger (see find_shape_target)
  int half_size;          // declare half the real data_size
};

// A wrong shape needs a dimension the model fixes: a dynamic dimension accepts any size. 0 if every dimension is dynamic (or rank 0).
static int find_shape_target(int *input, int *dim) {
  for (int i = 0; i < opt.num_inputs; i++)
    for (int d = 0; d < opt.inputs[i].rank; d++)
      if (!opt.inputs[i].dynamic[d]) return *input = i, *dim = d, 1;
  return 0;
}

static size_t request_bytes(void) {
  size_t total = 0;
  for (int i = 0; i < opt.num_inputs; i++) total += element_count(opt.inputs[i].shape, opt.inputs[i].rank) * opt.inputs[i].elem;
  return total ? total : 1;
}

static Tensors *build(int id, const struct Build *b) {
  int count = b->empty ? 0 : opt.num_inputs - (b->drop_last ? 1 : 0), target_input = -1, target_dim = 0;
  if (b->other_shape) find_shape_target(&target_input, &target_dim);
  Tensors *t = calloc(1, sizeof *t);
  t->id = id;
  t->num_tensors = count;
  t->tensors = calloc(count ? count : 1, sizeof *t->tensors);
  for (int i = 0; i < count; i++) {
    int dtype = opt.inputs[i].dtype, rank = opt.inputs[i].rank, shape[8];
    size_t elem = opt.inputs[i].elem;
    const char *name = opt.inputs[i].name;
    memcpy(shape, opt.inputs[i].shape, sizeof shape);
    if (i == 0 && b->first_name) name = b->first_name;
    if (i == 0 && b->other_dtype) dtype = dtype == 1 ? 2 : 1, elem = element_size(dtype);
    if (i == target_input) shape[target_dim] += 1;
    size_t bytes = element_count(shape, rank) * elem;
    TensorDescriptor *d = &t->tensors[i];
    d->name = strdup(name);
    d->data_type = dtype;
    d->rank = rank;
    d->shape = malloc(rank ? rank * sizeof(int) : 1);
    memcpy(d->shape, shape, rank * sizeof(int));
    d->data = calloc(1, bytes ? bytes : 1);
    fill(d->data, bytes, dtype);
    d->data_size = (i == 0 && b->half_size) ? bytes / 2 : bytes;
  }
  return t;
}

// Is a rejected request exactly as the harness built it? (The runtime must not touch or free it.)
static int untouched(const Tensors *t, int id, const struct Build *b) {
  Tensors *expected = build(id, b);
  int same = t->id == expected->id && t->num_tensors == expected->num_tensors;
  for (int i = 0; same && i < t->num_tensors; i++) {
    const TensorDescriptor *a = &t->tensors[i], *e = &expected->tensors[i];
    same = !strcmp(a->name, e->name) && a->data_type == e->data_type && a->rank == e->rank && a->data_size == e->data_size &&
           !memcmp(a->shape, e->shape, a->rank * sizeof(int)) && !memcmp(a->data, e->data, e->data_size);
  }
  free_tensors(expected);
  return same;
}

static uint64_t fnv(uint64_t h, const void *data, size_t n) {
  for (size_t i = 0; i < n; i++) h = (h ^ ((const unsigned char *)data)[i]) * 1099511628211ull;
  return h;
}

#define MAX_READ ((size_t)256 << 20)  // no conformance model returns more than this per tensor; a bigger claim is a lie we do not test by reading it

// What the shape of an output says it holds, in bytes; 0 when the descriptor cannot be trusted (bad rank/shape, unknown type, absurd size).
static size_t expected_bytes(const TensorDescriptor *d) {
  size_t elem = element_size(d->data_type);
  if (!elem || d->rank < 0 || d->rank > 8 || (d->rank > 0 && !d->shape)) return 0;
  size_t n = element_count(d->shape, d->rank);
  return n > MAX_READ / elem ? 0 : n * elem;
}

// The bytes of an output the harness may read: never more than the shape says (the runtime's data_size may be bigger).
static size_t readable_bytes(const TensorDescriptor *d) {
  size_t expected = expected_bytes(d);
  return d->data ? (d->data_size < expected ? d->data_size : expected) : 0;
}

// A Tensors whose descriptor array can be indexed at all.
static int tensors_ok(const Tensors *o) { return o->num_tensors >= 0 && (o->num_tensors == 0 || o->tensors); }
static int shown_tensors(const Tensors *o) { return !tensors_ok(o) ? 0 : o->num_tensors < 16 ? o->num_tensors : 16; }

// snprintf that never moves past the end of the buffer.
static void add(char *buf, size_t *n, size_t size, const char *fmt, ...) {
  if (*n + 1 >= size) return;
  va_list ap;
  va_start(ap, fmt);
  int written = vsnprintf(buf + *n, size - *n, fmt, ap);
  va_end(ap);
  if (written > 0) *n = *n + (size_t)written < size ? *n + (size_t)written : size - 1;
}

// What the runtime returned, as JSON members: routing, allocation, sizes, a hash of the data. Never reads more bytes than
// shape x element size says (a runtime that reports a bigger data_size must not crash the harness).
static void describe(const Tensors *o, char *buf, size_t size, uint64_t *hash_out) {
  size_t n = 0;
  const void *ptrs[96];
  int num_ptrs = 0, alloc_ok = tensors_ok(o), size_ok = tensors_ok(o);
  uint64_t hash = 1469598103934665603ull;
  add(buf, &n, size, "\"id\":%d,\"num\":%d,\"tensors\":[", o->id, o->num_tensors);
  for (int i = 0; i < shown_tensors(o); i++) {
    const TensorDescriptor *d = &o->tensors[i];
    int rank_ok = d->rank >= 0 && d->rank <= 8 && (d->rank == 0 || d->shape);
    size_t elem = element_size(d->data_type), expected = expected_bytes(d);
    alloc_ok &= d->name && d->rank >= 0 && (d->rank == 0 || d->shape) && (d->data_size == 0 || d->data);
    if (elem && rank_ok) size_ok &= d->data_size == expected;
    ptrs[num_ptrs++] = d->name, ptrs[num_ptrs++] = d->shape, ptrs[num_ptrs++] = d->data;
    hash = fnv(hash, d->data, readable_bytes(d));
    add(buf, &n, size, "%s{\"name\":", i ? "," : "");
    if (d->name) {
      add(buf, &n, size, "\"");
      for (const char *p = d->name; *p && n + 8 < size && p - d->name < 4096; p++) add(buf, &n, size, "%02x", (unsigned char)*p);
      add(buf, &n, size, "\"");
    } else add(buf, &n, size, "null");
    add(buf, &n, size, ",\"dtype\":%d,\"rank\":%d,\"shape\":[", d->data_type, d->rank);
    for (int k = 0; rank_ok && k < d->rank; k++) add(buf, &n, size, "%s%d", k ? "," : "", d->shape[k]);
    add(buf, &n, size, "],\"data_size\":%zu}", d->data_size);
  }
  int distinct = 1;
  for (int a = 0; a < num_ptrs; a++)
    for (int b = a + 1; b < num_ptrs; b++) distinct &= !(ptrs[a] && ptrs[a] == ptrs[b]);
  add(buf, &n, size, "],\"alloc_ok\":%s,\"size_ok\":%s,\"distinct\":%s,\"hash\":\"%016llx\"", alloc_ok ? "true" : "false",
      size_ok ? "true" : "false", distinct ? "true" : "false", (unsigned long long)hash);
  if (hash_out) *hash_out = hash;
}

static void dump_output(const char *step, const Tensors *o) {
  if (!opt.dump) return;
  for (int i = 0; i < shown_tensors(o); i++) {
    const TensorDescriptor *d = &o->tensors[i];
    char path[600];
    snprintf(path, sizeof path, "%s/%s_%d.bin", opt.dump, step, i);
    FILE *f = fopen(path, "wb");
    if (!f) continue;
    if (readable_bytes(d)) fwrite(d->data, 1, readable_bytes(d), f);  // never past what the shape says
    fclose(f);
  }
}

// ---- helpers shared by the scenarios ---------------------------------------------------------------------

static char big[1 << 16];

static int calibrated;

// The first answered request sets the pace: a slow device (or a slow first inference) must not turn into failures further on.
static void calibrate(double ms, int answered) {
  calibrated = 1;
  if (!answered) return;
  if (ms * 20 > opt.timeout_ms) opt.timeout_ms = (int)(ms * 20);
  if (ms * 10 > opt.lost_after_ms) opt.lost_after_ms = (int)(ms * 10);
}

// enqueue + retrieve one request; records both statuses, the routing and the output. Returns the retrieve status.
// The very first call may take up to --warmup-ms.
static int roundtrip(const char *step, int model_id, int id) {
  struct Build valid = {0};
  Tensors *input = build(id, &valid), *output = NULL;
  begin(step);
  RuntimeStatus enq = rt.enqueue_input(model_id, input);
  if (enq != RUNTIME_STATUS_SUCCESS) {
    free_tensors(input);
    char extra[64];
    snprintf(extra, sizeof extra, "\"enqueue\":%d", enq);
    done(step, NO_STATUS, NULL, extra);
    return enq;
  }
  int out_model = -1;
  double t0 = now_ms();
  RuntimeStatus got = rt.retrieve_output(&out_model, &output, calibrated ? opt.timeout_ms : opt.warmup_ms);
  double ms = now_ms() - t0;
  if (!calibrated) calibrate(ms, got == RUNTIME_STATUS_SUCCESS);
  char head[96];
  snprintf(head, sizeof head, "\"enqueue\":%d,\"model_id\":%d,\"ms\":%.2f", enq, out_model, ms);
  if (got == RUNTIME_STATUS_SUCCESS && output) {
    describe(output, big, sizeof big, NULL);
    char *extra = malloc(sizeof big + 128);
    snprintf(extra, sizeof big + 128, "%s,%s", head, big);
    done(step, got, NULL, extra);
    dump_output(step, output);
    free_tensors(output);
    free(extra);
  } else done(step, got, NULL, head);
  return got;
}

static void timed_retrieve(const char *step, int timeout_ms) {
  int model_id = -1;
  Tensors *output = NULL;
  begin(step);
  double t0 = now_ms();
  RuntimeStatus s = rt.retrieve_output(&model_id, &output, timeout_ms);
  char extra[96];
  snprintf(extra, sizeof extra, "\"model_id\":%d,\"ms\":%.1f,\"timeout\":%d", model_id, now_ms() - t0, timeout_ms);
  done(step, s, NULL, extra);
  if (s == RUNTIME_STATUS_SUCCESS) free_tensors(output);
}

static void start(void) {
  do_init("init", opt.config, 1);
  load_user_model("load");
  roundtrip("warmup", 0, 1);  // sets the timeouts for the rest of the scenario
}

static void info_snapshot(const char *step) { do_string(step, rt.get_info); }

// Every scenario ends with cleanup: a runtime that crashes at exit without it must not spoil the other checks.
static int finish(void) {
  do_cleanup("cleanup");
  return 0;
}

// ---- scenarios -------------------------------------------------------------------------------------------

// Retry after a failed load, load from memory, buffers copied, model ids.
int scenario_model_load(void) {
  do_init("init", opt.config, 1);
  ModelConfig missing = {.file_path = "/nonexistent/oaax-conformance.model"};
  begin("load_missing");
  done("load_missing", rt.load_models(1, &missing), NULL, NULL);
  ModelConfig real = {.file_path = opt.model};
  begin("load_retry");  // not load_user_model: a runtime that cannot load after a failed load is the finding here
  RuntimeStatus retry = rt.load_models(1, &real);
  done("load_retry", retry, NULL, NULL);
  if (retry == RUNTIME_STATUS_SUCCESS) roundtrip("infer_after_retry", 0, 1);
  do_cleanup("cleanup_retry");

  do_init("init_memory", opt.config, 1);
  FILE *f = fopen(opt.model, "rb");
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  rewind(f);
  unsigned char *data = malloc(size);
  if (fread(data, 1, size, f) != (size_t)size) size = 0;
  fclose(f);
  ModelConfig memory = {.model_data = data, .model_size = (size_t)size};
  begin("load_memory");
  RuntimeStatus s = rt.load_models(1, &memory);
  done("load_memory", s, NULL, NULL);
  memset(data, 0xAA, size);  // the runtime must have copied what it needs
  free(data);
  if (s == RUNTIME_STATUS_SUCCESS) roundtrip("infer_after_memory_load", 0, 2);
  do_cleanup("cleanup_memory");

  do_init("init_path", opt.config, 1);
  char *path = strdup(opt.model);
  ModelConfig by_path = {.file_path = path};
  begin("load_path_copy");
  s = rt.load_models(1, &by_path);
  done("load_path_copy", s, NULL, NULL);
  memset(path, 'X', strlen(path));
  free(path);
  if (s == RUNTIME_STATUS_SUCCESS) roundtrip("infer_after_path_overwritten", 0, 3);
  do_cleanup("cleanup_path");

  do_init("init_two", opt.config, 1);
  ModelConfig two[2] = {{.file_path = opt.model}, {.file_path = opt.model}};
  begin("load_two");
  s = rt.load_models(2, two);
  done("load_two", s, NULL, NULL);
  if (s == RUNTIME_STATUS_SUCCESS) {
    roundtrip("infer_model0", 0, 10);
    roundtrip("infer_model1", 1, 11);
  }
  do_cleanup("cleanup_two");
  return 0;
}

// One valid request: routing, allocation and sizes of the output (dumped with --dump).
int scenario_roundtrip(void) {
  start();
  roundtrip("roundtrip", 0, 4242);
  return finish();
}

int scenario_timeouts(void) {
  start();
  timed_retrieve("retrieve_0", 0);
  timed_retrieve("retrieve_200", 200);
  struct Build valid = {0};
  Tensors *input = build(1, &valid);
  begin("enqueue_one");
  RuntimeStatus s = rt.enqueue_input(0, input);
  done("enqueue_one", s, NULL, NULL);
  if (s != RUNTIME_STATUS_SUCCESS) free_tensors(input);
  timed_retrieve("retrieve_negative", -1);
  return finish();
}

// 8 prepared requests: time spent enqueueing against 8 requests run one after the other.
int scenario_enqueue_latency(void) {
  start();
  struct Build valid = {0};
  enum { N = 8 };
  Tensors *batch[N];
  for (int i = 0; i < N; i++) batch[i] = build(100 + i, &valid);
  int accepted = 0;
  double t0 = now_ms();
  for (int i = 0; i < N; i++) {
    if (rt.enqueue_input(0, batch[i]) == RUNTIME_STATUS_SUCCESS) accepted++;
    else free_tensors(batch[i]);
  }
  double enqueue_ms = now_ms() - t0;
  for (int i = 0; i < accepted; i++) {
    int model_id;
    Tensors *o = NULL;
    if (rt.retrieve_output(&model_id, &o, opt.timeout_ms) == RUNTIME_STATUS_SUCCESS) free_tensors(o);
  }
  t0 = now_ms();
  for (int i = 0; i < N; i++) roundtrip("sequential", 0, 200 + i);
  double sequential_ms = now_ms() - t0;
  char extra[160];
  snprintf(extra, sizeof extra, "\"accepted\":%d,\"enqueue_ms\":%.2f,\"sequential_ms\":%.2f", accepted, enqueue_ms, sequential_ms);
  begin("latency");
  done("latency", NO_STATUS, NULL, extra);
  return finish();
}

// One malformed request: record what enqueue said; if it was rejected, is the request untouched? if accepted, is it answered?
static void try_case(const char *step, int model_id, Tensors *input, const struct Build *b, int id, int null_input) {
  begin(step);
  RuntimeStatus s = rt.enqueue_input(model_id, null_input ? NULL : input);
  char extra[160];
  if (s != RUNTIME_STATUS_SUCCESS) {
    int intact = null_input || untouched(input, id, b);
    snprintf(extra, sizeof extra, "\"accepted\":false,\"intact\":%s", intact ? "true" : "false");
    done(step, s, NULL, extra);
    if (!null_input) free_tensors(input);
    return;
  }
  int out_model = -1, answered = 0;
  Tensors *o = NULL;
  if (rt.retrieve_output(&out_model, &o, opt.lost_after_ms) == RUNTIME_STATUS_SUCCESS) {  // later than this a request counts as lost
    answered = o->id == id;
    free_tensors(o);
  }
  snprintf(extra, sizeof extra, "\"accepted\":true,\"answered\":%s", answered ? "true" : "false");
  done(step, s, NULL, extra);
}

int scenario_validation(void) {
  start();
  struct Build valid = {0};
  try_case("invalid_id_negative", -1, build(1, &valid), &valid, 1, 0);
  try_case("invalid_id_high", 1, build(2, &valid), &valid, 2, 0);
  try_case("null_tensors", 0, NULL, &valid, 3, 1);
  struct Build empty = {.empty = 1}, none = {.drop_last = 1};
  try_case("no_tensors", 0, build(4, &empty), &empty, 4, 0);
  if (opt.num_inputs >= 2) try_case("missing_input", 0, build(5, &none), &none, 5, 0);
  else {
    begin("missing_input");
    done("missing_input", NO_STATUS, NULL, "\"skipped\":true");
  }
  struct Build dtype = {.other_dtype = 1}, shape = {.other_shape = 1}, half = {.half_size = 1};
  struct Build name = {.first_name = "oaax-conformance-unknown-input"};
  try_case("wrong_dtype", 0, build(6, &dtype), &dtype, 6, 0);
  int shape_input, shape_dim;
  if (find_shape_target(&shape_input, &shape_dim)) try_case("wrong_shape", 0, build(7, &shape), &shape, 7, 0);
  else {
    begin("wrong_shape");
    done("wrong_shape", NO_STATUS, NULL, "\"skipped\":true");  // every dimension is dynamic: no shape is wrong
  }
  try_case("short_data_size", 0, build(8, &half), &half, 8, 0);
  try_case("unknown_name", 0, build(9, &name), &name, 9, 0);
  roundtrip("still_usable", 0, 99);
  return finish();
}

// ---- concurrency -----------------------------------------------------------------------------------------

struct Producer {
  int index, count, num_models, enqueue_failures;
};

static void *produce(void *arg) {
  struct Producer *p = arg;
  struct Build valid = {0};
  for (int i = 0; i < p->count; i++) {
    Tensors *t = build(p->index * 100000 + i, &valid);
    if (rt.enqueue_input(i % p->num_models, t) != RUNTIME_STATUS_SUCCESS) {
      p->enqueue_failures++;
      free_tensors(t);
    }
  }
  return NULL;
}

// The outputs of one request, kept to compare later ones with.
struct Reference {
  int num, dtype[16];
  size_t size[16];
  void *data[16];
};

static size_t checked_bytes(const TensorDescriptor *d) { return readable_bytes(d); }

static void keep(struct Reference *r, const Tensors *o) {
  r->num = shown_tensors(o);
  for (int i = 0; i < r->num; i++) {
    r->dtype[i] = o->tensors[i].data_type;
    r->size[i] = checked_bytes(&o->tensors[i]);
    r->data[i] = malloc(r->size[i] ? r->size[i] : 1);
    memcpy(r->data[i], o->tensors[i].data, r->size[i]);
  }
}

// Same tensors, floats within 1e-4 relative (runtimes may sum in another order), everything else exactly.
static int matches(const struct Reference *r, const Tensors *o) {
  if (o->num_tensors != r->num || !tensors_ok(o)) return 0;
  for (int i = 0; i < r->num; i++) {
    const TensorDescriptor *d = &o->tensors[i];
    if ((int)d->data_type != r->dtype[i] || checked_bytes(d) != r->size[i]) return 0;
    if (r->dtype[i] != 1 && r->dtype[i] != 11) {
      if (memcmp(d->data, r->data[i], r->size[i])) return 0;
      continue;
    }
    size_t step = r->dtype[i] == 1 ? 4 : 8;
    for (size_t off = 0; off + step <= r->size[i]; off += step) {
      double a = r->dtype[i] == 1 ? *(const float *)((char *)r->data[i] + off) : *(const double *)((char *)r->data[i] + off);
      double b = r->dtype[i] == 1 ? *(const float *)((char *)d->data + off) : *(const double *)((char *)d->data + off);
      double diff = a > b ? a - b : b - a, scale = (a < 0 ? -a : a) > (b < 0 ? -b : b) ? (a < 0 ? -a : a) : (b < 0 ? -b : b);
      if (diff > 1e-6 && diff > 1e-4 * scale) return 0;
    }
  }
  return 1;
}

int scenario_concurrency(void) {
  do_init("init", opt.config, 1);
  ModelConfig two[2] = {{.file_path = opt.model}, {.file_path = opt.model}};
  begin("load");
  RuntimeStatus s = rt.load_models(2, two);
  done("load", s, NULL, NULL);
  int models = 2;
  if (s != RUNTIME_STATUS_SUCCESS) {  // a runtime that holds one model at a time still gets the concurrency tests, on model id 0
    ModelConfig one[1] = {{.file_path = opt.model}};
    begin("load_one");
    s = rt.load_models(1, one);
    done("load_one", s, NULL, NULL);
    models = 1;
  }
  if (s != RUNTIME_STATUS_SUCCESS) {
    end_marker("model");
    exit(EXIT_MODEL);
  }
  roundtrip("warmup", 0, 1);  // sets the timeouts

  struct Reference reference[2] = {{0}, {0}};
  for (int m = 0; m < models; m++) {  // the same input run alone is the reference for every model id
    struct Build valid = {0};
    Tensors *t = build(1, &valid), *o = NULL;
    int out_model;
    if (rt.enqueue_input(m, t) != RUNTIME_STATUS_SUCCESS) free_tensors(t);
    else if (rt.retrieve_output(&out_model, &o, opt.timeout_ms) == RUNTIME_STATUS_SUCCESS) {
      keep(&reference[m], o);
      free_tensors(o);
    }
  }

  int producers = opt.producers, per_producer = opt.per_producer;
  long cap = (long)((256u << 20) / request_bytes());  // never hold more than ~256 MB of requests at once
  if ((long)producers * per_producer > cap) per_producer = cap / producers > 1 ? (int)(cap / producers) : 1;
  int total = producers * per_producer;
  struct Producer *args = calloc(producers, sizeof *args);
  pthread_t *threads = calloc(producers, sizeof *threads);
  char *seen = calloc(total, 1);
  begin("run");
  for (int p = 0; p < producers; p++) {
    args[p] = (struct Producer){.index = p, .count = per_producer, .num_models = models};
    pthread_create(&threads[p], NULL, produce, &args[p]);
  }
  int received = 0, duplicates = 0, wrong_model = 0, mismatch = 0, unknown_id = 0, stalled = 0;
  double t0 = now_ms();
  while (received + duplicates < total) {
    int model_id = -1;
    Tensors *o = NULL;
    if (rt.retrieve_output(&model_id, &o, opt.timeout_ms) != RUNTIME_STATUS_SUCCESS) {
      stalled = 1;
      break;
    }
    int p = o->id / 100000, i = o->id % 100000;
    if (p < 0 || p >= producers || i >= per_producer) unknown_id++;
    else {
      if (seen[p * per_producer + i]++) duplicates++;
      else received++;
      wrong_model += model_id != i % models;
      mismatch += model_id >= 0 && model_id < models && reference[model_id].num && !matches(&reference[model_id], o);
    }
    free_tensors(o);
  }
  int enqueue_failures = 0;
  for (int p = 0; p < producers; p++) pthread_join(threads[p], NULL), enqueue_failures += args[p].enqueue_failures;
  char extra[480];
  snprintf(extra, sizeof extra,
           "\"models\":%d,\"expected\":%d,\"received\":%d,\"duplicates\":%d,\"wrong_model\":%d,\"mismatch\":%d,\"unknown_id\":%d,"
           "\"enqueue_failures\":%d,\"stalled\":%s,\"ms\":%.0f",
           models, total, received, duplicates, wrong_model, mismatch, unknown_id, enqueue_failures, stalled ? "true" : "false", now_ms() - t0);
  done("run", NO_STATUS, NULL, extra);
  do_cleanup("cleanup");
  return 0;
}

// ---- lifecycle under load --------------------------------------------------------------------------------

int scenario_cleanup_inflight(void) {
  start();
  struct Build valid = {0};
  int accepted = 0;
  for (int i = 0; i < 20; i++) {
    Tensors *t = build(300 + i, &valid);
    if (rt.enqueue_input(0, t) == RUNTIME_STATUS_SUCCESS) accepted++;
    else free_tensors(t);
  }
  char extra[64];
  snprintf(extra, sizeof extra, "\"accepted\":%d", accepted);
  begin("enqueued");
  done("enqueued", NO_STATUS, NULL, extra);
  double t0 = now_ms();
  begin("cleanup");
  RuntimeStatus s = rt.cleanup();
  snprintf(extra, sizeof extra, "\"ms\":%.0f", now_ms() - t0);
  done("cleanup", s, NULL, extra);
  do_init("init_again", opt.config, 0);
  load_user_model("load_again");
  roundtrip("infer_again", 0, 1);
  return finish();
}

static void *block_in_retrieve(void *arg) {
  int model_id = -1;
  Tensors *o = NULL;
  *(int *)arg = rt.retrieve_output(&model_id, &o, -1);
  if (*(int *)arg == RUNTIME_STATUS_SUCCESS) free_tensors(o);
  return NULL;
}

int scenario_cleanup_wakeup(void) {
  start();
  int status = -1;
  pthread_t t;
  pthread_create(&t, NULL, block_in_retrieve, &status);
  usleep(300 * 1000);
  do_cleanup("cleanup");
  struct timespec deadline;
  clock_gettime(CLOCK_REALTIME, &deadline);
  deadline.tv_sec += 5;
  int woke = pthread_timedjoin_np(t, NULL, &deadline) == 0;
  char extra[64];
  snprintf(extra, sizeof extra, "\"woke\":%s,\"retrieve_status\":%d", woke ? "true" : "false", status);
  begin("wakeup");
  done("wakeup", NO_STATUS, NULL, extra);
  if (!woke) {  // the thread is stuck inside the runtime: do not wait for it again
    end_marker("ok");
    _exit(0);
  }
  return 0;
}

int scenario_flood(void) {
  start();
  struct Build valid = {0};
  int accepted = 0, rejected = 0, sent = 0, requested = opt.flood;
  long cap = (long)((256u << 20) / request_bytes());  // requests wait inside the runtime: never more than ~256 MB of them
  if (requested > cap) requested = cap < 10 ? 10 : (int)cap;
  double max_ms = 0, t_all = now_ms();
  for (int i = 0; i < requested && !out_of_time(0); i++) {
    Tensors *t = build(1000 + i, &valid);
    double t0 = now_ms();
    RuntimeStatus s = rt.enqueue_input(0, t);
    double ms = now_ms() - t0;
    if (ms > max_ms) max_ms = ms;
    sent++;
    if (s == RUNTIME_STATUS_SUCCESS) accepted++;
    else rejected++, free_tensors(t);
  }
  double enqueue_total = now_ms() - t_all;
  char *seen = calloc(requested, 1);
  int answered = 0, duplicates = 0;
  for (int i = 0; i < accepted; i++) {
    int model_id;
    Tensors *o = NULL;
    if (rt.retrieve_output(&model_id, &o, opt.timeout_ms) != RUNTIME_STATUS_SUCCESS) break;
    int idx = o->id - 1000;
    if (idx >= 0 && idx < requested && seen[idx]++) duplicates++;
    else answered++;
    free_tensors(o);
  }
  char extra[260];
  snprintf(extra, sizeof extra,
           "\"requests\":%d,\"requested\":%d,\"accepted\":%d,\"rejected\":%d,\"answered\":%d,\"duplicates\":%d,\"max_enqueue_ms\":%.1f,\"enqueue_total_ms\":%.0f",
           sent, opt.flood, accepted, rejected, answered, duplicates, max_ms, enqueue_total);
  begin("flood");
  done("flood", NO_STATUS, NULL, extra);
  return finish();
}

int scenario_info_state(void) {
  do_init("init", opt.config, 1);
  info_snapshot("info_no_model");
  load_user_model("load");
  roundtrip("warmup", 0, 1);
  info_snapshot("info_loaded");
  struct Build valid = {0};
  for (int i = 0; i < 3; i++) {
    Tensors *t = build(50 + i, &valid);
    if (rt.enqueue_input(0, t) != RUNTIME_STATUS_SUCCESS) free_tensors(t);
  }
  info_snapshot("info_in_flight");
  for (int i = 0; i < 3; i++) {
    int model_id;
    Tensors *o = NULL;
    if (rt.retrieve_output(&model_id, &o, opt.timeout_ms) == RUNTIME_STATUS_SUCCESS) free_tensors(o);
  }
  info_snapshot("info_idle");
  return finish();
}

int scenario_model_cycles(void) {
  start();  // a runtime that cannot start or load the model at all is "blocked", not a cycle failure
  do_cleanup("cleanup_first");
  int baseline = thread_count(), failures = 0, completed = 0;
  long rss_first = -1;
  char first_failure[128] = "";
  for (int c = 0; c < opt.model_cycles && !failures && !out_of_time(0); c++) {
    begin("model_cycle");
    ModelConfig model = {.file_path = opt.model};
    RuntimeStatus si = rt.init(opt.config), sl = si ? si : rt.load_models(1, &model);
    int bad = si || sl;
    struct Build valid = {0};
    for (int i = 0; !bad && i < 10; i++) {
      Tensors *t = build(i, &valid), *o = NULL;
      int model_id;
      if (rt.enqueue_input(0, t) != RUNTIME_STATUS_SUCCESS) free_tensors(t), bad = 1;
      else if (rt.retrieve_output(&model_id, &o, opt.timeout_ms) != RUNTIME_STATUS_SUCCESS) bad = 1;
      else free_tensors(o);
    }
    RuntimeStatus sc = si ? RUNTIME_STATUS_SUCCESS : rt.cleanup();
    if (bad || sc) failures++, snprintf(first_failure, sizeof first_failure, "cycle %d: init=%d load=%d cleanup=%d", c, si, sl, sc);
    else completed++;
    if (c == 0) rss_first = rss_bytes();
  }
  settle(baseline, opt.settle_ms);
  char extra[340];
  snprintf(extra, sizeof extra,
           "\"cycles\":%d,\"completed\":%d,\"failures\":%d,\"first_failure\":\"%s\",\"rss_first\":%ld,\"rss_last\":%ld,\"threads_before\":%d,\"threads_after\":%d",
           opt.model_cycles, completed, failures, first_failure, rss_first, rss_bytes(), baseline, thread_count());
  done("model_cycles", NO_STATUS, NULL, extra);
  return 0;
}

int scenario_leak_requests(void) {
  start();
  struct Build valid = {0};
  int failures = 0, completed = 0;
  long rss_before = 0;
  for (int i = 0; i < 50 + opt.requests && !out_of_time(0); i++) {
    if (i == 50) rss_before = rss_bytes();  // the first requests grow caches and pools
    Tensors *t = build(i, &valid), *o = NULL;
    int model_id;
    if (rt.enqueue_input(0, t) != RUNTIME_STATUS_SUCCESS) free_tensors(t), failures++;
    else if (rt.retrieve_output(&model_id, &o, opt.timeout_ms) != RUNTIME_STATUS_SUCCESS) failures++;
    else free_tensors(o);
    if (i >= 50) completed++;
  }
  char extra[200];
  snprintf(extra, sizeof extra, "\"requests\":%d,\"completed\":%d,\"failures\":%d,\"rss_before\":%ld,\"rss_after\":%ld", opt.requests, completed, failures,
           rss_before, rss_bytes());
  begin("leak");
  done("leak", NO_STATUS, NULL, extra);
  return finish();
}
