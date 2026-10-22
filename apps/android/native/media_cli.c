/* Run the packaged FFmpeg CLI in its own process, preserving ordinary pipes,
 * cancellation and argv semantics. Executables live in Android's read-only
 * nativeLibraryDir, where Android permits execution on API 29 and later. */
#include <dlfcn.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    char library[PATH_MAX];
    if (!realpath(argv[0], library)) { perror("media executable"); return 126; }
    char *slash = strrchr(library, '/');
    if (!slash) return 126;
    strcpy(slash + 1, "libffmpegkit.so");
    // The bundle's constructor prints its build banner on stdout. Redirect it
    // before dlopen so JSON probes and raw BGR pipes stay byte-clean.
    int output_fd = dup(STDOUT_FILENO);
    if (output_fd < 0) return 126;
    fflush(stdout);
    dup2(STDERR_FILENO, STDOUT_FILENO);
    void *handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    fflush(stdout);
    dup2(output_fd, STDOUT_FILENO);
    close(output_fd);
    if (!handle) { fprintf(stderr, "%s\n", dlerror()); return 126; }
    int probe = strstr(argv[0], "ffprobe") != NULL;
    const char *tool = probe ? "ffprobe" : "ffmpeg";
    char symbol[80];
    snprintf(symbol, sizeof(symbol), "%s_init_argv", tool);
    void *(*init)(int, const char *const *) = dlsym(handle, symbol);
    snprintf(symbol, sizeof(symbol), "%s_run", tool);
    int (*run)(void *) = dlsym(handle, symbol);
    snprintf(symbol, sizeof(symbol), "%s_free", tool);
    void (*cleanup)(void *) = dlsym(handle, symbol);
    void (*set_log)(void *) = dlsym(handle, "av_log_set_callback");
    if (!init || !run || !cleanup || !set_log) { fprintf(stderr, "Incomplete media runtime\n"); return 126; }
    set_log(dlsym(handle, "av_log_default_callback"));
    argv[0] = (char *)tool;
    void *ctx = init(argc, (const char *const *)argv);
    if (!ctx) return 2;
    int code = run(ctx);
    if (probe) {
        char *(*output)(void *) = dlsym(handle, "ffprobe_get_output");
        if (output) {
            char *text = output(ctx);
            if (text) { fputs(text, stdout); free(text); }
        }
    }
    cleanup(ctx);
    fflush(NULL);
    // Library destructors also print to stdout. All owned work has completed;
    // let the OS release the process without those unrelated diagnostic lines.
    _exit(code == 0 ? 0 : 1);
}
