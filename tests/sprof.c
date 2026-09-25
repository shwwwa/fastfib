/* sampling profiler: runs fibonacci(n) `reps` times while a sampler thread
   suspends every other thread ~every 1 ms and records its instruction pointer.
   usage: sprof n reps  -> writes raw addresses to sprof_out.txt */
#define UNIT_TEST_NO_MAIN
#include "../fastfib.c"
#include <tlhelp32.h>
#include <mmsystem.h>

static volatile LONG g_stop;
static DWORD g_self_tid, g_sampler_tid;
static uint64_t *g_samples;
static size_t g_nsamples, g_cap = 1 << 24;

static DWORD WINAPI sampler(LPVOID arg) {
    (void)arg;
    DWORD pid = GetCurrentProcessId();
    HANDLE hs[64];
    int main_idx[64], nh = 0;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    THREADENTRY32 te = {.dwSize = sizeof te};
    for (BOOL ok = Thread32First(snap, &te); ok && nh < 64; ok = Thread32Next(snap, &te)) {
        if (te.th32OwnerProcessID != pid || te.th32ThreadID == g_sampler_tid) continue;
        HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, FALSE, te.th32ThreadID);
        if (h) main_idx[nh] = te.th32ThreadID == g_self_tid, hs[nh++] = h;
    }
    CloseHandle(snap);
    while (!g_stop) {
        for (int i = 0; i < nh; i++) {
            if (SuspendThread(hs[i]) == (DWORD)-1) continue;
            CONTEXT ctx = {.ContextFlags = CONTEXT_CONTROL};
            if (GetThreadContext(hs[i], &ctx) && g_nsamples < g_cap)
                g_samples[g_nsamples++] = ((uint64_t)main_idx[i] << 63) | ctx.Rip;
            ResumeThread(hs[i]);
        }
        if (g_nsamples < g_cap) g_samples[g_nsamples++] = 0; /* end of round */
        Sleep(1);
    }
    return 0;
}
int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s n reps\n", argv[0]);
        return 2;
    }
    uint64_t n = strtoull(argv[1], NULL, 10);
    int reps = atoi(argv[2]);
    g_samples = malloc(g_cap * sizeof(uint64_t));
    g_self_tid = GetCurrentThreadId();
    Big w = fibonacci(n); /* warm-up: tables, pool, arena */
    big_free(&w);
    timeBeginPeriod(1);
    HANDLE th = CreateThread(NULL, 0, sampler, NULL, 0, &g_sampler_tid);
    for (int r = 0; r < reps; r++) {
        Big x = fibonacci(n);
        big_free(&x);
    }
    g_stop = 1;
    WaitForSingleObject(th, INFINITE);
    FILE *f = fopen("sprof_out.txt", "w");
    fprintf(f, "base %llx\n", (unsigned long long)(uintptr_t)GetModuleHandleA(NULL));
    for (size_t i = 0; i < g_nsamples; i++) fprintf(f, "%llx\n", (unsigned long long)g_samples[i]);
    fclose(f);
    /* outside the exe: which DLL, and the nearest export for msvcrt/ntdll */
    char modname[MAX_PATH];
    struct { char name[64]; size_t n; } mods[32];
    int nm = 0;
    uintptr_t exe = (uintptr_t)GetModuleHandleA(NULL);
    for (size_t i = 0; i < g_nsamples; i++) {
        uintptr_t rip = (uintptr_t)(g_samples[i] & ~(1ull << 63));
        if (!rip) continue;
        HMODULE hm;
        if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)rip, &hm)) strcpy(modname, "?");
        else if ((uintptr_t)hm == exe) continue;
        else { GetModuleFileNameA(hm, modname, MAX_PATH); }
        char *base = strrchr(modname, '\\'); base = base ? base + 1 : modname;
        int k = 0;
        while (k < nm && strcmp(mods[k].name, base)) k++;
        if (k == nm && nm < 32) { strncpy(mods[nm].name, base, 63); mods[nm].name[63] = 0; mods[nm++].n = 0; }
        if (k < 32) mods[k].n++;
    }
    for (int k = 0; k < nm; k++) printf("  outside: %-24s %zu\n", mods[k].name, mods[k].n);
    printf("%zu samples\n", g_nsamples);
    return 0;
}
