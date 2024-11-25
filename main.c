#include <SDL.h>
#include <stdio.h>
#include <stdbool.h>

#define WINDOW_WIDTH 1200
#define WINDOW_HEIGHT 900

double scale_number(double cv, double min, double max, double nmin, double nmax){
    double range1 = max - min;
    double range2 = nmax - nmin;

    double new_number = ((cv * range2)/range1) + (nmin);
    return new_number;
}

int choose_brightness_mandelblot(int i, int j, float minsize, float maxsize);
void create_mandelbrot(SDL_Renderer * renderer, float scale);

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
    /**
     * Function guide:
     * SDL_SetRenderDrawColor changes the color of the pixel to be drawn
     * SDL_RenderDrawPoint draws the color
     */

    // Clear screen (black background) for now
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    create_mandelbrot(renderer, -2);
    // //places in the middle
    // for(int i = 0; i < WINDOW_WIDTH; i++){
    //     for(int j = 0; j < WINDOW_HEIGHT; j++){
    //         int brightness = choose_brightness_mandelblot(i, j, -1, 1);
            
    //         // Choose white color
    //         SDL_SetRenderDrawColor(renderer, brightness, brightness, brightness, 255);
    //         SDL_RenderDrawPoint(renderer, i, j);

    //     }//end for
    // }//end for
    // //SDL_RenderDrawPoint(renderer, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);

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



/**
 * making brightness on mandelblot
 */
int choose_brightness_mandelblot(int i, int j, float minsize, float maxsize){
    double a = scale_number((double)i, 0, WINDOW_WIDTH, minsize, maxsize);
    double b = scale_number((double)j, 0, WINDOW_HEIGHT, minsize, maxsize);


    int iterations = 0; 
    int maxiterations = 50;
    int isInfinite = 16;
    double ca = a; // constant
    double cb = b; //constant

    //for each pixel, check if it's infinite or not
    while (iterations < maxiterations){
        double aa = a * a - b * b;
        double bb = 2 * a * b;

        a = aa + ca;
        b = bb + cb;

        double check_bounds = a + b;
        //convert to absolute
        if(check_bounds < 0){
            check_bounds *= -1;
        }

        if(check_bounds > isInfinite){
            break;
        }
        iterations++;
    }//end while

    //mapping function

    double scaled = scale_number((double)iterations, 0, maxiterations, 0, 1);
    int brightness = scale_number(sqrt(scaled), 0, 1, 0, 255);

    //if in mandelbrot make color black. 
    if(iterations >= maxiterations){
        brightness = 0;
    }
    return brightness;
}

void create_mandelbrot(SDL_Renderer * renderer, float scale){

    //places in the middle
    for(int i = 0; i < WINDOW_WIDTH; i++){
        for(int j = 0; j < WINDOW_HEIGHT; j++){
            int brightness = choose_brightness_mandelblot(i, j, scale * -1, scale);
            
            // Choose white color
            SDL_SetRenderDrawColor(renderer, brightness, brightness, brightness, 255);
            SDL_RenderDrawPoint(renderer, i, j);

        }//end for
    }//end for

}//end create_Mandelbrot

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




/**
 * Notes from Ellie
 * Get vectors instead of pixels
 * pixels are rastor data – assigning points on a screen
 * 
 * Vectorized data – a different kind of line that comes out of adobe
 * - when scaled up, it looks just as high definition
 * - it scales distances rather than specific points. 
 */