// Operating-system specifics of the harness: see platform.h.
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <tlhelp32.h>
// ============================================================ Windows
double now_ms(void) {
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}

void sleep_ms(int ms) { Sleep((DWORD)ms); }

static char dl_error[1024];

void *plat_dlopen(const char *path) {
  char full[PATH_MAX];
  if (!_fullpath(full, path, sizeof full)) snprintf(full, sizeof full, "%s", path);
  HMODULE m = LoadLibraryExA(full, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);  // the library's own folder is searched for its dependencies
  if (m) return m;
  DWORD code = GetLastError();
  char msg[512] = "";
  FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, code, 0, msg, sizeof msg, NULL);
  msg[strcspn(msg, "\r\n")] = 0;
  // 126 (module not found) and 1114 (a DLL init routine failed) are usually a DEPENDENCY: the message must not start with the file's own name
  int own = code != 126 && code != 1114;
  snprintf(dl_error, sizeof dl_error, "%s%s%s (Windows error %lu)", own ? path : "", own ? ": " : "", msg, (unsigned long)code);
  return NULL;
}

void *plat_dlsym(void *lib, const char *name) { return (void *)GetProcAddress((HMODULE)lib, name); }
int plat_dlclose(void *lib) { return FreeLibrary((HMODULE)lib) ? 0 : 1; }
const char *plat_dlerror(void) { return dl_error; }

int plat_is_mapped(const char *path) {
  char full[PATH_MAX];
  if (!_fullpath(full, path, sizeof full)) return 0;
  return GetModuleHandleA(full) != NULL;
}

typedef struct { plat_thread_fn fn; void *arg; } ThreadStart;
static DWORD WINAPI thread_main(LPVOID p) {
  ThreadStart s = *(ThreadStart *)p;
  free(p);
  s.fn(s.arg);
  return 0;
}

int plat_thread_start(plat_thread *t, plat_thread_fn fn, void *arg) {
  ThreadStart *s = malloc(sizeof *s);
  s->fn = fn, s->arg = arg;
  *t = CreateThread(NULL, 0, thread_main, s, 0, NULL);
  return *t ? 0 : 1;
}

void plat_thread_join(plat_thread t) {
  WaitForSingleObject(t, INFINITE);
  CloseHandle(t);
}

int plat_thread_join_timeout(plat_thread t, int ms) {
  if (WaitForSingleObject(t, (DWORD)ms) != WAIT_OBJECT_0) return 0;
  CloseHandle(t);
  return 1;
}

void plat_hard_exit(int code) { TerminateProcess(GetCurrentProcess(), (UINT)code); }

int plat_temp_file(char *path, size_t size, const char *prefix) {
  char dir[MAX_PATH], name[MAX_PATH];
  if (!GetTempPathA(sizeof dir, dir) || !GetTempFileNameA(dir, prefix, 0, name)) return -1;  // creates the file
  snprintf(path, size, "%s", name);
  return 0;
}

char *plat_realpath(const char *path) { return _fullpath(NULL, path, 0); }

typedef HRESULT(WINAPI *get_description_fn)(HANDLE, PWSTR *);

// The Toolhelp snapshot lists every thread of the machine: keep this process's.
static int walk_threads(char *buf, size_t size, size_t *n_out) {
  HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
  if (snap == INVALID_HANDLE_VALUE) return 0;
  get_description_fn get_description = (get_description_fn)(void *)GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetThreadDescription");  // Windows 10 1607+
  THREADENTRY32 e = {.dwSize = sizeof e};
  size_t n = *n_out;
  int count = 0, first = 1;
  for (BOOL ok = Thread32First(snap, &e); ok; ok = Thread32Next(snap, &e)) {
    if (e.th32OwnerProcessID != GetCurrentProcessId()) continue;
    count++;
    if (!buf || n + 140 >= size) continue;
    char name[130] = "";
    HANDLE h = get_description ? OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, e.th32ThreadID) : NULL;
    PWSTR wide = NULL;
    if (h && SUCCEEDED(get_description(h, &wide)) && wide) {
      WideCharToMultiByte(CP_UTF8, 0, wide, -1, name, sizeof name - 1, NULL, NULL);  // UTF-16 on the wire: always valid UTF-8 here
      LocalFree(wide);
    }
    if (h) CloseHandle(h);
    n += snprintf(buf + n, size - n, "%s\"", first ? "" : ",");
    for (const char *p = name; *p && n + 4 < size; p++) n += snprintf(buf + n, size - n, "%02x", (unsigned char)*p);
    n += snprintf(buf + n, size - n, "\"");
    first = 0;
  }
  CloseHandle(snap);
  *n_out = n;
  return count;
}

#include <psapi.h>
long long rss_bytes(void) {  // private bytes (commit): the working set is trimmed by the OS at will, which would look like a leak that shrank
  PROCESS_MEMORY_COUNTERS_EX c = {.cb = sizeof c};
  return GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&c, sizeof c) ? (long long)c.PrivateUsage : -1;
}

void thread_names(char *buf, size_t size) {
  size_t n = snprintf(buf, size, "[");
  walk_threads(buf, size, &n);
  snprintf(buf + n, size - n, "]");
}

int thread_count(void) {
  size_t n = 0;
  return walk_threads(NULL, 0, &n);
}

#else
#include <dirent.h>
// ============================================================ POSIX
double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

void sleep_ms(int ms) {
  struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
  nanosleep(&ts, NULL);
}

void *plat_dlopen(const char *path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
void *plat_dlsym(void *lib, const char *name) { return dlsym(lib, name); }
int plat_dlclose(void *lib) { return dlclose(lib); }
const char *plat_dlerror(void) { return dlerror(); }

// Is the file listed in /proc/self/maps?
int plat_is_mapped(const char *path) {
  char *resolved = realpath(path, NULL);
  char line[PATH_MAX + 256];
  int mapped = 0;
  FILE *maps = fopen("/proc/self/maps", "r");
  while (maps && resolved && fgets(line, sizeof line, maps)) mapped |= strstr(line, resolved) != NULL;
  if (maps) fclose(maps);
  free(resolved);
  return mapped;
}

int plat_thread_start(plat_thread *t, plat_thread_fn fn, void *arg) { return pthread_create(t, NULL, fn, arg); }
void plat_thread_join(plat_thread t) { pthread_join(t, NULL); }

int plat_thread_join_timeout(plat_thread t, int ms) {
  struct timespec deadline;
  clock_gettime(CLOCK_REALTIME, &deadline);
  deadline.tv_sec += ms / 1000;
  deadline.tv_nsec += (ms % 1000) * 1000000L;
  if (deadline.tv_nsec >= 1000000000L) deadline.tv_sec++, deadline.tv_nsec -= 1000000000L;
  return pthread_timedjoin_np(t, NULL, &deadline) == 0;
}

void plat_hard_exit(int code) { _exit(code); }

int plat_temp_file(char *path, size_t size, const char *prefix) {
  snprintf(path, size, "/tmp/%s-XXXXXX", prefix);
  int fd = mkstemp(path);
  if (fd < 0) return -1;
  close(fd);
  return 0;
}

char *plat_realpath(const char *path) { return realpath(path, NULL); }

long long rss_bytes(void) {
  long pages = 0, resident = 0;
  FILE *f = fopen("/proc/self/statm", "r");
  if (!f) return -1;
  if (fscanf(f, "%ld %ld", &pages, &resident) != 2) resident = -1;
  fclose(f);
  return resident < 0 ? -1 : (long long)resident * sysconf(_SC_PAGESIZE);
}

// Thread names (/proc/self/task/*/comm) as a JSON array.
void thread_names(char *buf, size_t size) {
  size_t n = 0;
  n += snprintf(buf + n, size - n, "[");
  DIR *dir = opendir("/proc/self/task");
  struct dirent *e;
  int first = 1;
  while (dir && (e = readdir(dir)) && n + 140 < size) {
    if (e->d_name[0] == '.') continue;
    char path[320], name[64] = "?";
    snprintf(path, sizeof path, "/proc/self/task/%s/comm", e->d_name);
    FILE *f = fopen(path, "r");
    if (f) {
      if (fgets(name, sizeof name, f)) name[strcspn(name, "\n")] = 0;
      fclose(f);
    }
    n += snprintf(buf + n, size - n, "%s\"", first ? "" : ",");
    for (const char *p = name; *p && n + 4 < size; p++) n += snprintf(buf + n, size - n, "%02x", (unsigned char)*p);  // thread names are bytes
    n += snprintf(buf + n, size - n, "\"");
    first = 0;
  }
  if (dir) closedir(dir);
  snprintf(buf + n, size - n, "]");
}

int thread_count(void) {
  char buf[8192];
  thread_names(buf, sizeof buf);
  int n = 0;
  for (char *p = buf; *p; p++) n += *p == '"';
  return n / 2;
}
#endif
