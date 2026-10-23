// Entry point: parses the command line, loads the library, runs the scenario.
#include "common.h"
#include "scenarios.h"

const struct { const char *name; int (*run)(void); } SCENARIOS[] = {
    {"static", scenario_static},
    {"lifecycle", scenario_lifecycle},
    {"cycles", scenario_cycles},
    {"unload", scenario_unload},
    {"exit-without-cleanup", scenario_exit_without_cleanup},
    {"errors", scenario_errors},
    {"null-pointers", scenario_null_pointers},
    {"load-twice", scenario_load_twice},
    {"model-load", scenario_model_load},
    {"roundtrip", scenario_roundtrip},
    {"timeouts", scenario_timeouts},
    {"enqueue-latency", scenario_enqueue_latency},
    {"validation", scenario_validation},
    {"concurrency", scenario_concurrency},
    {"cleanup-inflight", scenario_cleanup_inflight},
    {"cleanup-wakeup", scenario_cleanup_wakeup},
    {"flood", scenario_flood},
    {"info-state", scenario_info_state},
    {"model-cycles", scenario_model_cycles},
    {"leak-requests", scenario_leak_requests},
};


// ---- main ---------------------------------------------------------------------------------------------------

static int usage(void) {
  fprintf(stderr, "usage: oaax-harness <scenario> <library> --out FILE [-c key=value]... [-m MODEL] [-i name:DTYPE:dims]... [--budget-ms N] [--cycles N] [--settle-ms N] [--dump DIR] [...]\n");
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
    else if (!strcmp(argv[i], "--dump") && has_value) opt.dump = argv[++i];
    else if (!strcmp(argv[i], "--flood") && has_value) opt.flood = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--requests") && has_value) opt.requests = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--producers") && has_value) opt.producers = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--per-producer") && has_value) opt.per_producer = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--model-cycles") && has_value) opt.model_cycles = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--timeout-ms") && has_value) opt.timeout_ms = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--lost-after-ms") && has_value) opt.lost_after_ms = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--budget-ms") && has_value) opt.budget_ms = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warmup-ms") && has_value) opt.warmup_ms = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-i") && has_value) {
      if (opt.num_inputs >= 16 || !parse_input(argv[++i], &opt.inputs[opt.num_inputs++])) return usage();
    }
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
  out_of_time(0);  // starts the clock of the time budget
  rc = load_library();
  if (rc) {
    end_marker(rc == EXIT_DLOPEN ? "dlopen" : "symbols");
    return rc;
  }
  for (size_t i = 0; i < sizeof SCENARIOS / sizeof SCENARIOS[0]; i++)
    if (!strcmp(opt.scenario, SCENARIOS[i].name)) {
      rc = SCENARIOS[i].run();
      end_marker("ok");
      return rc;
    }
  fprintf(stderr, "unknown scenario: %s\n", opt.scenario);
  return EXIT_USAGE;
}
