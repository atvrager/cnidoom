/*
 * syscalls.c — Minimal libc stubs for bare-metal RISC-V.
 *
 * Provides just enough to satisfy Doom's libc needs:
 *   - _sbrk: heap allocation (used by malloc)
 *   - _write: output to UART (used by printf)
 *   - _read, _close, _fstat, _lseek, _isatty: stubs
 *   - _exit: halt
 *   - File I/O via RISC-V semihosting (fopen, fread, fseek, fclose)
 *
 * Semihosting protocol for RISC-V:
 *   a0 = operation number
 *   a1 = pointer to parameter block
 *   Trigger via: .insn i SYSTEM, 0, x0, x0, 0x003B (ebreak variant)
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>

/* Semihosting operation numbers. */
#define SYS_OPEN 0x01
#define SYS_CLOSE 0x02
#define SYS_WRITEC 0x03
#define SYS_WRITE0 0x04
#define SYS_WRITE 0x05
#define SYS_READ 0x06
#define SYS_READC 0x07
#define SYS_SEEK 0x0A
#define SYS_FLEN 0x0C
#define SYS_EXIT 0x18

/*
 * Issue a semihosting call.
 * RISC-V semihosting uses the sequence: slli x0, x0, 0x1f; ebreak; srai x0, x0,
 * 7 This magic sequence is recognized by QEMU.
 */
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
/* Heap (sbrk)                                                        */
/* ------------------------------------------------------------------ */

extern char __heap_start;
extern char __heap_end;

static char* heap_ptr = 0;

void* _sbrk(ptrdiff_t incr) {
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
/* Console I/O (UART)                                                 */
/* ------------------------------------------------------------------ */

extern void uart_putc(char c);

/*
 * Semihosting fd offset: semihosting can return handles 0, 1, 2 which
 * collide with newlib's stdin/stdout/stderr.  We add SH_FD_OFFSET when
 * returning from _open and subtract it in every other syscall.
 */
#define SH_FD_OFFSET 3

int _write(int fd, const char* buf, int len) {
  if (fd <= 2) {
    /* stdout/stderr → UART. */
    for (int i = 0; i < len; i++) {
      if (buf[i] == '\n') uart_putc('\r');
      uart_putc(buf[i]);
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
  /* File read via semihosting. */
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
  /*
   * Map POSIX open flags to semihosting fopen-style mode.
   *
   * Semihosting modes (per ARM DUI 0471):
   *   0 = "r"    1 = "rb"    2 = "r+"   3 = "r+b"
   *   4 = "w"    5 = "wb"    6 = "w+"   7 = "w+b"
   *   8 = "a"    9 = "ab"   10 = "a+"  11 = "a+b"
   *
   * All modes are binary on bare-metal (no text-mode translation),
   * so we always use the 'b' variants.
   */
  int sh_mode;
  int accmode = flags & 3; /* O_RDONLY=0, O_WRONLY=1, O_RDWR=2 */
  if (flags & 0x400 /* O_APPEND */) {
    sh_mode = (accmode == 0) ? 9 : 11; /* "ab" or "a+b" */
  } else if (flags & 0x40 /* O_CREAT */) {
    sh_mode = (accmode == 2) ? 7 : 5; /* "w+b" or "wb" */
  } else {
    sh_mode = (accmode == 0) ? 1 : 3; /* "rb" or "r+b" */
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
  if (fd <= 2) return 0; /* Don't close stdin/stdout/stderr */
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
    /* Get file length, then seek to length + offset. */
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
  /* SEEK_CUR not directly supported by semihosting — not used by Doom. */
  errno = EINVAL;
  return -1;
}

int _fstat(int fd, struct stat* st) {
  memset(st, 0, sizeof(*st));
  if (fd <= 2) {
    st->st_mode = S_IFCHR; /* Character device for stdin/stdout/stderr */
  } else {
    int sh_fd = fd - SH_FD_OFFSET;
    st->st_mode = S_IFREG; /* Regular file */
    /* Get file length via semihosting. */
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
  long args[2] = {0x20026 /* ADP_Stopped_ApplicationExit */, status};
  semihosting_call(SYS_EXIT, (long)args);
  for (;;) {
    __asm__ volatile("wfi");
  }
}

/* ------------------------------------------------------------------ */
/* kill / getpid stubs (required by some newlib configurations)       */
/* ------------------------------------------------------------------ */

int _kill(int pid, int sig) {
  (void)pid;
  (void)sig;
  errno = EINVAL;
  return -1;
}

int _getpid(void) { return 1; }

/* mkdir / link / unlink stubs (Doom calls mkdir for savegames). */
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
