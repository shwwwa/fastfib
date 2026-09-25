/* reads hex numbers (one per line) from stdin, prints big_to_string of each */
#define UNIT_TEST_NO_MAIN
#include "../fastfib.c"
static int hexval(int c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; }
int main(void) {
    static char line[1 << 26];
    while (fgets(line, sizeof line, stdin)) {
        size_t len = strcspn(line, "\r\n");
        size_t n = (len + 15) / 16;
        Big a;
        a.d = xcalloc(n ? n : 1, sizeof(uint64_t));
        for (size_t i = 0; i < len; i++) { /* digit i from the right */
            size_t k = len - 1 - i;
            a.d[k / 16] |= (uint64_t)hexval(line[i]) << (4 * (k % 16));
        }
        a.n = normalize_len(a.d, n ? n : 1);
        char *s = big_to_string(&a);
        puts(s);
        free(s);
        big_free(&a);
    }
    return 0;
}
