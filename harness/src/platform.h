// The only place that knows the operating system: dynamic loading, threads, sleeping, and what a process can tell about itself.
// Linux/macOS: dlfcn + pthreads + /proc.  Windows: LoadLibrary + Win32 threads + Toolhelp.  Everything else includes this and stays portable.
#pragma once
#include <stddef.h>

#ifdef _WIN32
#define PLAT_WINDOWS 1
#define _CRT_SECURE_NO_WARNINGS 1
#include <windows.h>
#include <direct.h>
#include <io.h>
#define strdup _strdup
#define unlink _unlink
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
typedef HANDLE plat_thread;
#else
#define PLAT_WINDOWS 0
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
typedef pthread_t plat_thread;
#endif

typedef void *(*plat_thread_fn)(void *);

double now_ms(void);                       // monotonic
void sleep_ms(int ms);
void *plat_dlopen(const char *path);       // NULL on failure: plat_dlerror() says why (it names the file or the dependency)
void *plat_dlsym(void *lib, const char *name);
int plat_dlclose(void *lib);               // 0 = ok
const char *plat_dlerror(void);
int plat_is_mapped(const char *path);      // is this library file still loaded in the process?
int plat_thread_start(plat_thread *t, plat_thread_fn fn, void *arg);  // 0 = ok
void plat_thread_join(plat_thread t);
int plat_thread_join_timeout(plat_thread t, int ms);                  // 1 = joined, 0 = still running
void plat_hard_exit(int code);             // exit without running destructors or atexit handlers
int plat_temp_file(char *path, size_t size, const char *prefix);      // creates an empty file, writes its path; -1 on error
char *plat_realpath(const char *path);     // malloc'd absolute path or NULL
void thread_names(char *buf, size_t size); // names of this process's threads as a JSON array of hex strings
int thread_count(void);
long long rss_bytes(void);              // resident (Linux) / private (Windows) bytes of this process, -1 when unknown
