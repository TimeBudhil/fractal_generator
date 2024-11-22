#include <SDL.h>
#include <stdio.h>
#include <stdbool.h>

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600


//clang main.c -I/opt/homebrew/include/SDL2 -L/opt/homebrew/lib -lSDL2 -o game
int main(int argc, char* argv[]) {

    // make sure we don't have any command line arguments.
    if (argc > 1) {
        printf("Error: This program doesn't accept any command line arguments\n");
        printf("Usage: %s\n", argv[0]);
        return 1;
    }

    //initialize the window
    SDL_Window* window = NULL;

    //initialize renderer
    SDL_Renderer* renderer = NULL;

    //initialize boolean for loop, for now we aren't quitting, so to false
    bool quit = false;

    //initialize the event in SDL.
    SDL_Event event;

    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL initialization failed! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }

    // Create window
    window = SDL_CreateWindow("Fractal",
                            SDL_WINDOWPOS_CENTERED,
                            SDL_WINDOWPOS_CENTERED,
                            WINDOW_WIDTH, WINDOW_HEIGHT,
                            SDL_WINDOW_SHOWN);

    //error check for initializing the window
    if (!window) {
        printf("Window creation failed! SDL_Error: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }



    /** Renderer used to draw pixels */
    // Create renderer
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        printf("Renderer creation failed! SDL_Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    /**Main functions */

    // Clear screen (black background) for now
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    // Draw white pixel in center
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

    //places in the middle
    SDL_RenderDrawPoint(renderer, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);

    // Update screen
    SDL_RenderPresent(renderer);

/*loop for quitting the window*/
    // Main loop
    while (!quit) {
        // Handle events
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = true;
            } else if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    quit = true;
                }
            }
        }
    }

    // Cleanup for the window, renderer, and SDL
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

/**Previous attempts of coding */
    // /**Creating Textrues
    //  * First initialize,
    //  * 
    //  * Then draw to the textures
    //  * 
    //  * Then use those textures to draw different colors
    //  */
    // auto red_texture = SDL_CreateTexture(renderer, 
    // SDL_PIXEL_FORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, WINDOW_WIDTH,WINDOW_HEIGHT);
    // SDL_SetRenderTarget(renderer, red_texture); // select target
    // SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); //set color to read
    // SDL_RenderClear(renderer);

    // //  blue texture
    // auto blue_texture = SDL_CreateTexture(renderer, 
    // SDL_PIXEL_FORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, WINDOW_WIDTH,WINDOW_HEIGHT);
    // SDL_SetRenderTarget(renderer, blue_texture); // select target
    // SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255); //set color to read
    // SDL_RenderClear(renderer);
    
    // /**Show textures for screen */
    // SDL_SetRenderTarget(renderer, nullptr);

    // // Show red texture
    // SDL_RenderCopy(renderer, red_texture, nullptr, nullptr);
    // SDL_RendererPresent(renderer);
    // SDL_Delay(10000);

    // // show blue textures
    // SDL_RenderCopy(renderer, red_texture, nullptr, nullptr);
    // SDL_RendererPresent(renderer);
    // SDL_Delay(10000);



