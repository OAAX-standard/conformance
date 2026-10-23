#pragma once
// Every scenario returns the process exit code; see EXIT_* in common.h.
int scenario_static(void);
int scenario_lifecycle(void);
int scenario_cycles(void);
int scenario_unload(void);
int scenario_exit_without_cleanup(void);
int scenario_errors(void);
int scenario_null_pointers(void);
int scenario_load_twice(void);

// Model tier (need -m MODEL and -i for every model input)
int scenario_model_load(void);
int scenario_roundtrip(void);
int scenario_timeouts(void);
int scenario_enqueue_latency(void);
int scenario_validation(void);
int scenario_concurrency(void);
int scenario_cleanup_inflight(void);
int scenario_cleanup_wakeup(void);
int scenario_flood(void);
int scenario_info_state(void);
int scenario_model_cycles(void);
int scenario_leak_requests(void);

int parse_input(const char *arg, struct InputSpec *spec);
