/*
 * frame_viewer.c — Host-side SDL framebuffer viewer (fallback).
 *
 * Reads raw 640×400 XRGB8888 frames from stdin and displays them
 * in an SDL window.  Used as a fallback if ramfb doesn't work:
 *
 *   qemu-system-riscv32 -semihosting -nographic -kernel doom.elf \
 *     | ./frame_viewer
 *
 * The guest would use semihosting SYS_WRITE to pipe frames to stdout.
 *
 * Build (host):
 *   cc -o frame_viewer frame_viewer.c $(sdl2-config --cflags --libs)
 */

#include <SDL.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 640
#define HEIGHT 400
#define FRAME_SIZE (WIDTH * HEIGHT * 4) /* XRGB8888 */

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
    return 1;
  }

  SDL_Window* window =
      SDL_CreateWindow("Doom RV32 Agent", SDL_WINDOWPOS_CENTERED,
                       SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
  if (!window) {
    fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
    return 1;
  }

  SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);
  SDL_Texture* texture =
      SDL_CreateTexture(renderer, SDL_PIXELFORMAT_XRGB8888,
                        SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

  unsigned char* frame = malloc(FRAME_SIZE);
  if (!frame) {
    fprintf(stderr, "malloc failed\n");
    return 1;
  }

  int running = 1;
  while (running) {
    /* Read one frame from stdin. */
    size_t total = 0;
    while (total < FRAME_SIZE) {
      size_t n = fread(frame + total, 1, FRAME_SIZE - total, stdin);
      if (n == 0) {
        running = 0;
        break;
      }
      total += n;
    }
    if (!running) break;

    /* Upload to texture and render. */
    SDL_UpdateTexture(texture, NULL, frame, WIDTH * 4);
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);

    /* Poll events (so the window stays responsive). */
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        running = 0;
      }
    }
  }

  free(frame);
  SDL_DestroyTexture(texture);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}
