/*
 * cnidoom_syscalls.c — Minimal libc stubs for the cnidoom library.
 *
 * Refactored from platform/rv32/syscalls.c.  Key change: _write for
 * fd≤2 calls cnidoom_putc() instead of uart_putc(), making console
 * output portable.  Both _write and _sbrk are weak so downstream
 * users can override them.
 *
 * Semihosting file I/O (_open, _read, _close, _lseek, _fstat) and
 * stubs (_kill, _getpid, mkdir, _exit) are unchanged.
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>

#include "cnidoom.h"

/* Semihosting operation numbers. */
#define SYS_OPEN 0x01
#define SYS_CLOSE 0x02
#define SYS_WRITE 0x05
#define SYS_READ 0x06
#define SYS_SEEK 0x0A
#define SYS_FLEN 0x0C
#define SYS_EXIT 0x18

static inline long semihosting_call(long op, long arg) {
  register long a0 __asm__("a0") = op;
  register long a1 __asm__("a1") = arg;
  __asm__ volatile(
      ".option push\n"
      ".option norvc\n"
      "slli x0, x0, 0x1f\n"
      "ebreak\n"
      "srai x0, x0, 7\n"
      ".option pop\n"
      : "+r"(a0)
      : "r"(a1)
      : "memory");
  return a0;
}

/* ------------------------------------------------------------------ */
/* Heap (sbrk) — weak so user can override                            */
/* ------------------------------------------------------------------ */

extern char __heap_start;
extern char __heap_end;

static char* heap_ptr = 0;

__attribute__((weak)) void* _sbrk(ptrdiff_t incr) {
  if (heap_ptr == 0) {
    heap_ptr = &__heap_start;
  }
  char* prev = heap_ptr;
  if (heap_ptr + incr > &__heap_end) {
    errno = ENOMEM;
    return (void*)-1;
  }
  heap_ptr += incr;
  return prev;
}

/* ------------------------------------------------------------------ */
/* Console I/O — uses cnidoom_putc() for portability                  */
/* ------------------------------------------------------------------ */

#define SH_FD_OFFSET 3

__attribute__((weak)) int _write(int fd, const char* buf, int len) {
  if (fd <= 2) {
    /* stdout/stderr → cnidoom_putc (platform callback). */
    for (int i = 0; i < len; i++) {
      if (buf[i] == '\n') cnidoom_putc('\r');
      cnidoom_putc(buf[i]);
    }
    return len;
  }
  /* File write via semihosting. */
  int sh_fd = fd - SH_FD_OFFSET;
  long args[3] = {sh_fd, (long)buf, len};
  long not_written = semihosting_call(SYS_WRITE, (long)args);
  return len - (int)not_written;
}

int _read(int fd, char* buf, int len) {
  if (fd <= 2) {
    return 0; /* No stdin on bare-metal. */
  }
  int sh_fd = fd - SH_FD_OFFSET;
  long args[3] = {sh_fd, (long)buf, len};
  long not_read = semihosting_call(SYS_READ, (long)args);
  if (not_read < 0) {
    errno = EIO;
    return -1;
  }
  return len - (int)not_read;
}

/* ------------------------------------------------------------------ */
/* File I/O via semihosting                                           */
/* ------------------------------------------------------------------ */

int _open(const char* path, int flags, int mode) {
  (void)mode;
  int sh_mode;
  int accmode = flags & 3;
  if (flags & 0x400 /* O_APPEND */) {
    sh_mode = (accmode == 0) ? 9 : 11;
  } else if (flags & 0x40 /* O_CREAT */) {
    sh_mode = (accmode == 2) ? 7 : 5;
  } else {
    sh_mode = (accmode == 0) ? 1 : 3;
  }
  long args[3] = {(long)path, sh_mode, (long)strlen(path)};
  long result = semihosting_call(SYS_OPEN, (long)args);
  if (result == -1) {
    errno = ENOENT;
    return -1;
  }
  return (int)result + SH_FD_OFFSET;
}

int _close(int fd) {
  if (fd <= 2) return 0;
  int sh_fd = fd - SH_FD_OFFSET;
  long args[1] = {sh_fd};
  semihosting_call(SYS_CLOSE, (long)args);
  return 0;
}

int _lseek(int fd, int offset, int whence) {
  int sh_fd = fd - SH_FD_OFFSET;
  if (whence == 0 /* SEEK_SET */) {
    long args[2] = {sh_fd, offset};
    long result = semihosting_call(SYS_SEEK, (long)args);
    if (result != 0) {
      errno = EIO;
      return -1;
    }
    return offset;
  }
  if (whence == 2 /* SEEK_END */) {
    long flen_args[1] = {sh_fd};
    long flen = semihosting_call(SYS_FLEN, (long)flen_args);
    if (flen < 0) {
      errno = EIO;
      return -1;
    }
    long target = flen + offset;
    long seek_args[2] = {sh_fd, target};
    long result = semihosting_call(SYS_SEEK, (long)seek_args);
    if (result != 0) {
      errno = EIO;
      return -1;
    }
    return (int)target;
  }
  errno = EINVAL;
  return -1;
}

int _fstat(int fd, struct stat* st) {
  memset(st, 0, sizeof(*st));
  if (fd <= 2) {
    st->st_mode = S_IFCHR;
  } else {
    int sh_fd = fd - SH_FD_OFFSET;
    st->st_mode = S_IFREG;
    long args[1] = {sh_fd};
    long flen = semihosting_call(SYS_FLEN, (long)args);
    if (flen >= 0) {
      st->st_size = flen;
    }
  }
  return 0;
}

int _isatty(int fd) { return (fd <= 2) ? 1 : 0; }

void _exit(int status) {
  long reason = (status == 0) ? 0x20026L : 0x20024L;
  semihosting_call(SYS_EXIT, reason);
  for (;;) {
    __asm__ volatile("wfi");
  }
}

/* ------------------------------------------------------------------ */
/* Stubs                                                              */
/* ------------------------------------------------------------------ */

int _kill(int pid, int sig) {
  (void)pid;
  (void)sig;
  errno = EINVAL;
  return -1;
}

int _getpid(void) { return 1; }

#include <sys/types.h>
int mkdir(const char* path, mode_t mode) {
  (void)path;
  (void)mode;
  errno = ENOENT;
  return -1;
}

int _link(const char* old, const char* new_) {
  (void)old;
  (void)new_;
  errno = EMLINK;
  return -1;
}

int _unlink(const char* path) {
  (void)path;
  errno = ENOENT;
  return -1;
}
