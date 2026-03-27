/*
 * qemu_main.c — Minimal entry point for the QEMU virt target.
 *
 * Links against libcnidoom.a and runs with default QEMU virt config.
 */

#include "cnidoom.h"

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  cnidoom_run(NULL);
  return 0;
}
