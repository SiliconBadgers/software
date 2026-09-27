// Force-included into the harness sources on Windows so profile.cpp needs no Windows-specific code
// and stays the same program as the one that produced the recorded 2026-09-22 baseline
// (tests/test_harness_equivalence.py enforces this). POSIX setenv() does not exist in the
// Windows CRT; _putenv_s sets the same process environment variable that getenv() reads.
#pragma once
#ifdef _WIN32
#include <stdlib.h>
static inline int setenv(const char * name, const char * value, int /*overwrite*/) {
    return _putenv_s(name, value);
}
#endif
