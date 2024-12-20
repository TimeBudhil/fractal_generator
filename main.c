#include <SDL.h> // include SDL<-- this must be linked properly in whatever programming is made
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

/** consider these functions:
 * https://discourse.libsdl.org/t/proposal-vector-graphics-api/27938
 */

/**
 * Numeric limit – doubles have a max representation
 * 
 * pixels vs vectors 
 */

//adjust window width and height to your wishes. 
//The smaller the window, the more easily the pages will load. 
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600


// the number of iterations to be run per individual pixel
//if you zoom in increase the cap of these iterations
// zoomed in coordinate conversion just another coordinate conversion
int maxIterations = 100;  // the higher this value is the, more "concrete" the fractal will look

// Initialize the variables to click the screen to move and zoom in or out.
int mouseClickX = -1;
int mouseClickY = -1;


// Global for modifying specifications and parameters of the fractals
double centerX = 0.0;  // how much right or left we go.  (multiply against zoomscale to be proportional) on complex plane
double centerY = 0.0;  //how much up or down we go (multiply against zoomscale to be proportional) on complex plane
double zoomScale = 1.0;  // Zoom scale factor
double baseWidth = WINDOW_WIDTH / 100;  // Base width of the view in complex plane
double baseHeight = WINDOW_HEIGHT / 100;  // Base width of the view in complex plane
double keyboardSpeed = 0.1;  // Speed of scrolling
int which = 1; //which set are we using?
double minimumReal;
double maximumReal;
double maximumComplex;
double minimumComplex;
double constantA = 0;// real constant ("constant" in terms of the julia equation) portion of the julia set.
double constantB = 0; //complex constant ("constant" in terms of the julia equation) portion of the julia set
double isInfinite = 25;

/**
 * Scalenumber–maps a current value index of a one-dimensional field onto a larger field.
 * eg: 9.5 on a 0-10 scale would be 95 on a 0-100 scale; the inputs would be scale_number(9.5, 0, 10, 0, 100);
 * 
 * @param cv the current value that we want to transform
 * @param min the current scale (one-dimensional) minimum
 * @param max the current scale (one-dimensional) maximum
 * @param nmin the transformed scale (one-dimensional) minimum
 * @param nmax the transformed scale (one-dimensional) maximum
 */
double scale_number(double cv, double min, double max, double nmin, double nmax);

/**
 * max – returns the highest value between a and b
 * @param a the first integer input
 * @param b second integer input
 */
int max(int a, int b);

/**
 * For one pixel, choose the brightness of that pixel based on the mandelbrot set's formula
 * @param renderer, pointer to the renderer of SDL which is where the pixel will be drawn (within the function)
 * @param i, the x coordinate of the pixel (within the resolution of the window–WINDOW_WIDTH and WINDOW_HEIGHT)
 * @param j, the y coordinate of the pixel (within the resolution of the window–WINDOW_WIDTH and WINDOW_HEIGHT)
 * @param minsize, scaling factor of the mandelbrot–based on how zoomed in the current image is or not
 * @param maxsize, scaling factor of the mandelbrot–based on how zoomed in the current image is or not
 * 
 * Quick explanation of scaling: The wider apart @param minsize and @param maxsize are, the more zoomed out
 * the image will be. If minsize + maxsize = 0, the image will be centered. If it's negative, the image will
 * be shifted right, and if positive, shifted left. 
 * 
 * Note: this is ONE pixel in one image of the mandelbrot–i.e. this function will be called WINDOW_WIDTH * WINDOW_HEIGHT
 * times.
 * 
 * This function draws the pixels according to the mandelbrot formula: z^2 + c;
 * The product of z^2 + c is applied into z, so the next iteration of z^2 + c = (z^2 + c)^2 + c
 * This is on the complex plane, so the values iterate and sometimes reach infinity, and sometimes don't. 
 * That is what the  MAX_ITERATIONS macro means–how many times we iterate this fomrula until it's too much. 
 * Technically, this can go on infinitely. But we cap it at 50, other tutorials cap this at 16 (the coding challenge)
 * Depending on the value and graphing this on the complex plane, coloring the values that result in a cycling towards infinity one color,
 * and coloring the other pixels white, we get the mandelbrot set fractal. The Julia set has a very similar iterative process, which
 * can have values that can be adjusted. 
 * 
 * This can be easily parallelized. For each zoom in, a new mandelbrot image is generated. 
 * This version of the function is only black and white
 */
void choose_brightness_mandelbrot(SDL_Renderer * renderer, int i, int j);


/**
 * For one pixel, choose the brightness of that pixel based on the mandelbrot set's formula
 * @param renderer, pointer to the renderer of SDL which is where the pixel will be drawn (within the function)
 * @param i, the x coordinate of the pixel (within the resolution of the window–WINDOW_WIDTH and WINDOW_HEIGHT)
 * @param j, the y coordinate of the pixel (within the resolution of the window–WINDOW_WIDTH and WINDOW_HEIGHT)
 * @param minsize, scaling factor of the mandelbrot–based on how zoomed in the current image is or not
 * @param maxsize, scaling factor of the mandelbrot–based on how zoomed in the current image is or not
 * 
 * Quick explanation of scaling: The wider apart @param minsize and @param maxsize are, the more zoomed out
 * the image will be. If minsize + maxsize = 0, the image will be centered. If it's negative, the image will
 * be shifted right, and if positive, shifted left. 
 * 
 * Note: this is ONE pixel in one image of the mandelbrot–i.e. this function will be called WINDOW_WIDTH * WINDOW_HEIGHT
 * times.
 * 
 * This function draws the pixels according to the mandelbrot formula: z^2 + c;
 * The product of z^2 + c is applied into z, so the next iteration of z^2 + c = (z^2 + c)^2 + c
 * This is on the complex plane, so the values iterate and sometimes reach infinity, and sometimes don't. 
 * That is what the  MAX_ITERATIONS macro means–how many times we iterate this fomrula until it's too much. 
 * Technically, this can go on infinitely. But we cap it at 50, other tutorials cap this at 16 (the coding challenge)
 * Depending on the value and graphing this on the complex plane, coloring the values that result in a cycling towards infinity one color,
 * and coloring the other pixels white, we get the mandelbrot set fractal. The Julia set has a very similar iterative process, which
 * can have values that can be adjusted. 
 * 
 * This can be easily parallelized. For each zoom in, a new mandelbrot image is generated. 
 * This version of the function is colored
 */
void choose_colorful_mandelbrot(SDL_Renderer * renderer, int i, int j);

/**
 * For one pixel, choose the brightness of that pixel based on the julia set's formula
 * @param renderer, pointer to the renderer of SDL which is where the pixel will be drawn (within the function)
 * @param i, the x coordinate of the pixel (within the resolution of the window–WINDOW_WIDTH and WINDOW_HEIGHT)
 * @param j, the y coordinate of the pixel (within the resolution of the window–WINDOW_WIDTH and WINDOW_HEIGHT)
 * 
 * Quick explanation of scaling: The wider apart @param minsize and @param maxsize are, the more zoomed out
 * the image will be. If minsize + maxsize = 0, the image will be centered. If it's negative, the image will
 * be shifted right, and if positive, shifted left. 
 * 
 * Note: this is ONE pixel in one image of the mandelbrot–i.e. this function will be called WINDOW_WIDTH * WINDOW_HEIGHT
 * times.
 * 
 * This function draws the pixels according to the mandelbrot formula: z^2 + c;
 * The product of z^2 + c is applied into z, so the next iteration of z^2 + c = (z^2 + c)^2 + c
 * This is on the complex plane, so the values iterate and sometimes reach infinity, and sometimes don't. 
 * That is what the  MAX_ITERATIONS macro means–how many times we iterate this fomrula until it's too much. 
 * Technically, this can go on infinitely. But we cap it at 50, other tutorials cap this at 16 (the coding challenge)
 * Depending on the value and graphing this on the complex plane, coloring the values that result in a cycling towards infinity one color,
 * and coloring the other pixels white, we get the mandelbrot set fractal. The Julia set has a very similar iterative process, which
 * can have values that can be adjusted. 
 * 
 * This can be easily parallelized. For each zoom in, a new mandelbrot image is generated. 
 * This version of the function is colored
 */
void choose_colorful_julia(SDL_Renderer * renderer, int i, int j);

/**
 * For one pixel, choose the brightness of that pixel based on the mandelbrot set's formula
 * @param renderer, pointer to the renderer of SDL which is where the pixel will be drawn (within the function)
 * @param i, the x coordinate of the pixel (within the resolution of the window–WINDOW_WIDTH and WINDOW_HEIGHT)
 * @param j, the y coordinate of the pixel (within the resolution of the window–WINDOW_WIDTH and WINDOW_HEIGHT)
 * @param minsize, scaling factor of the mandelbrot–based on how zoomed in the current image is or not
 * @param maxsize, scaling factor of the mandelbrot–based on how zoomed in the current image is or not
 * 
 * Quick explanation of scaling: The wider apart @param minsize and @param maxsize are, the more zoomed out
 * the image will be. If minsize + maxsize = 0, the image will be centered. If it's negative, the image will
 * be shifted right, and if positive, shifted left. 
 * 
 * Note: this is ONE pixel in one image of the mandelbrot–i.e. this function will be called WINDOW_WIDTH * WINDOW_HEIGHT
 * times.
 * 
 * This function draws the pixels according to the mandelbrot formula: z^2 + c;
 * The product of z^2 + c is applied into z, so the next iteration of z^2 + c = (z^2 + c)^2 + c
 * This is on the complex plane, so the values iterate and sometimes reach infinity, and sometimes don't. 
 * That is what the  MAX_ITERATIONS macro means–how many times we iterate this fomrula until it's too much. 
 * Technically, this can go on infinitely. But we cap it at 50, other tutorials cap this at 16 (the coding challenge)
 * Depending on the value and graphing this on the complex plane, coloring the values that result in a cycling towards infinity one color,
 * and coloring the other pixels white, we get the mandelbrot set fractal. The Julia set has a very similar iterative process, which
 * can have values that can be adjusted. 
 * 
 * This can be easily parallelized. For each zoom in, a new mandelbrot image is generated. 
 * This version of the function is based on the heatmap function–@KEVIN INSERT SOURCE HERE
 * that bases the color on the function heatmap..
 */
void choose_heatmap_mandelbrot(SDL_Renderer * renderer, int i, int j);

/**
 * Iteratively calls @function choose_brightness_mandelbrot or @function choose_heatmap_mandelbrot based on
 * the choice 
 * @param which – decides which fractal we will use
 * @param renderer – the renderer in SDL which is where the mandelbrot image will be drawn
 * @param scale – the scale parameter which is adjusted in MAIN based on the SDL scaling, will be updated as it is passed into this function
 */
void create_mandelbrot(SDL_Renderer * renderer, int which);

/**
 * choose_color
 * chooses color based on the proportion of min and max
 */
SDL_Color hsbToRgb(float hue, float saturation, float brightness);

/**Function heatmap takes a point and determines the points color based on its location 
 * @kevin please commment!, and add this not only to this header call but also to the actual function below
 */
void heatmap(float minimum, float maximum, float val, int* r, int* g, int* b);


/**taken from the interwebs */
//void save_texture(SDL_Renderer *ren, SDL_Texture *tex, const char *filename);
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

    //initialize the event in SDL.
    SDL_Event event;

    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        // print an error if we cannot print
        printf("SDL initialization failed! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }

    // Create window in SDL
    window = SDL_CreateWindow("Fractal", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);

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
        //error checking the renderer
        printf("Renderer creation failed, SDL_Error: %s\n", SDL_GetError());
        //clean up if the creation failed
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;//return errror
    }

    /**Main functions */
    /**
     * Function guide for understanding SDL:
     * SDL_SetRenderDrawColor changes the color of the pixel to be drawn
     * SDL_RenderDrawPoint draws the color
     */

    // Clear screen (black background) in the beginning
    //clearing the renderer
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer); 


    /* MAIN loop the event handling and creation of the mandelbrot*/
    //initialize boolean while loop, for now we aren't quitting, so to false
    bool quit = false;
    while (!quit) {

        //handle events using the SDL event systems
        /**CONTROLS:
         * 1. zooming in & zooming out: zooming with scroller; moving right left up down based on right left up down keys
         * 2. moving right and left:
         * 3. (not implemented, for now can be manually done) changing mandelbrot image: 1, 2, 3, 4 to choose between:
         *      1–mandelbrot based on brightness values
         *      2–heatmap mandelbrot
         *      3–(not implemented) Julia set with different controls
         *      4–(not implemented) Colorful Julia set... (add more)
         *      Other options: 5- an autoscrolling infite zooming fractal 
         * 4. (not implemented) Mouse controls–hold to zoom into that location, click to zoom in minorly, 
         * 5. (not implemented) other parameters – colors?
         * 
         */
        while (SDL_PollEvent(&event)) {

            /** If exit the window, the window closes and SDL_QUIT is true! */
            if (event.type == SDL_QUIT) {
                //set quit to true to exit the MAIN loop
                quit = true;

                /*Mouse operations–clicking and holding*/
            } else if (event.type == SDL_MOUSEBUTTONDOWN) { 
                // When mouse is clicked, capture the click position
                if (event.button.button == SDL_BUTTON_LEFT) { 
                    /**Ideally, we would want
                     * the clicking of the button to zoom into that location specifically
                     * secondly, we would want holding the click in that location to be a zooming into that location. 
                     */

                    int x, y;
                    SDL_GetMouseState(&x, &y);
                    printf("Mouse clicked at (%d, %d)\n", x, y);
                    centerX = scale_number(x, 0, WINDOW_HEIGHT, minimumReal, maximumReal);
                    centerY = scale_number(y, 0, WINDOW_HEIGHT, minimumComplex, maximumComplex);
            
                    zoomScale *= 0.9;
                }//end if

                /*Zooming in and Out*/
            } else if (event.type == SDL_KEYDOWN) {
                /**Ideally, the scrolling would move the screen in and out
                 * secondly, a scroll right or left would move the screen right or left
                 */

                /** moving right and left based on keyboard typing 
                 * Keys are: left, right, up, down, r to reset, and esc to secape
                */
                switch (event.key.keysym.sym) {
                    /* escape quits everything*/
                    case SDLK_ESCAPE:
                        quit = true;
                        break;
                    case SDLK_LEFT:
                        // Scroll left (move view to the left)
                        centerX -= keyboardSpeed * zoomScale;
                        break;
                    case SDLK_RIGHT:
                        // Scroll right (move view to the right)
                        centerX += keyboardSpeed * zoomScale;
                        break;
                    case SDLK_UP:
                        // Scroll up (move view up)
                        centerY += keyboardSpeed * zoomScale;
                        break;
                    case SDLK_DOWN:
                        // Scroll down (move view down)
                        centerY -= keyboardSpeed * zoomScale;
                        break;
                    case SDLK_EQUALS:
                        zoomScale *= 0.9;
                        break;
                    case SDLK_MINUS:
                        zoomScale /= 0.9;
                        break;
                    case SDLK_i:
                        maxIterations++;
                        break;
                    case SDLK_o:
                        maxIterations--;
                        break;
                    case SDLK_n: //n for negative
                        zoomScale *= -1;
                        break;
                    case SDLK_g:
                        isInfinite++;
                        break;
                    case SDLK_h:
                        isInfinite--;
                        break;
                        /*Adjust parameters on julia set*/
                    case SDLK_j:
                        constantA += 0.01;
                        break;
                    case SDLK_k:
                        constantA -= 0.01;
                        break;
                    case SDLK_s:
                        constantB += 0.01;
                        break;
                    case SDLK_d:
                        constantB -= 0.01;
                        break;

                    /* F for preset julia set*/
                    case SDLK_f:
                        constantA = -0.835;
                        constantB = -0.2321;
                        break;
                    case SDLK_1: 
                        which = 1;
                        printf("which: %d\n", which);
                        break;
                    case SDLK_2: 
                        which = 2;
                        printf("which: %d\n", which);
                        break;
                    case SDLK_3: 
                        which = 3;
                        printf("which: %d\n", which);
                        break;
                    case SDLK_4: 
                        which = 4;
                        printf("which: %d\n", which);
                        break;
                    //case SDLK_s: 
                        //save_texture(renderer,)
                        //break;
                    case SDLK_r:
                        // Reset to original view
                        centerX = 0.0;
                        centerY = 0.0;
                        zoomScale = 1.0;
                        break;
                }

                /**Scrolling based on teh scale */
            } else if (event.type == SDL_MOUSEWHEEL) {
                // Zoom logic (same as before)
                if (event.wheel.y > 0) { // Scroll up (zoom in)
                    zoomScale *= 0.9;  // Zoom in by reducing scale
                } else if (event.wheel.y < 0) { // Scroll down (zoom out)
                    zoomScale /= 0.9;  // Zoom out by increasing scale
                } 
                
            } else if (event.type == SDL_MOUSEMOTION) {
                // // Optional: Pan functionality with right mouse button
                // if (event.motion.state & SDL_BUTTON_RMASK) {
                //     // Right mouse button dragging to pan
                //     double panX = event.motion.xrel * (baseWidth * zoomScale) / WINDOW_WIDTH;
                //     double panY = event.motion.yrel * (baseHeight * zoomScale) / WINDOW_HEIGHT;
                //     centerX -= panX;
                //     centerY -= panY;
                // }
            }
        }

        // Clear screen
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // Modify create_mandelbrot to use new zoom parameters
        create_mandelbrot(renderer, which);

        // Update screen
        SDL_RenderPresent(renderer);
    }//end main while

    // Cleanup for the window, renderer, and SDL
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

/**
 * For one pixel, choose the brightness of that pixel based on the mandelbrot set's formula
 * @param renderer, pointer to the renderer of SDL which is where the pixel will be drawn (within the function)
 * @param i, the x coordinate of the pixel (within the resolution of the window–WINDOW_WIDTH and WINDOW_HEIGHT)
 * @param j, the y coordinate of the pixel (within the resolution of the window–WINDOW_WIDTH and WINDOW_HEIGHT)
 * @param minsize, scaling factor of the mandelbrot–based on how zoomed in the current image is or not
 * @param maxsize, scaling factor of the mandelbrot–based on how zoomed in the current image is or not
 * 
 * Quick explanation of scaling: The wider apart @param minsize and @param maxsize are, the more zoomed out
 * the image will be. If minsize + maxsize = 0, the image will be centered. If it's negative, the image will
 * be shifted right, and if positive, shifted left. 
 * 
 * Note: this is ONE pixel in one image of the mandelbrot–i.e. this function will be called WINDOW_WIDTH * WINDOW_HEIGHT
 * times.
 * 
 * This function draws the pixels according to the mandelbrot formula: z^2 + c;
 * The product of z^2 + c is applied into z, so the next iteration of z^2 + c = (z^2 + c)^2 + c
 * This is on the complex plane, so the values iterate and sometimes reach infinity, and sometimes don't. 
 * That is what the  MAX_ITERATIONS macro means–how many times we iterate this fomrula until it's too much. 
 * Technically, this can go on infinitely. But we cap it at 50, other tutorials cap this at 16 (the coding challenge)
 * Depending on the value and graphing this on the complex plane, coloring the values that result in a cycling towards infinity one color,
 * and coloring the other pixels white, we get the mandelbrot set fractal. The Julia set has a very similar iterative process, which
 * can have values that can be adjusted. 
 * 
 * This can be easily parallelized. For each zoom in, a new mandelbrot image is generated. 
 * This version of the function is only black and white
 */
void choose_brightness_mandelbrot(SDL_Renderer * renderer, int i, int j){
    double a = scale_number((double)i, 0, WINDOW_WIDTH, minimumReal, maximumReal); //real
    double b = scale_number((double)j, 0, WINDOW_HEIGHT, minimumComplex, maximumComplex); //complex

    int iterations = 0; 
    int isInfinite = 25;
    double ca = a; // constant
    double cb = b; //constant

    //for each pixel, check if it's infinite or not
    while (iterations < maxIterations){
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
    double scaled = scale_number((double)iterations, 0, maxIterations, 0, 1);
    int brightness = scale_number(sqrt(scaled), 0, 1, 0, 255);

    //if in mandelbrot make color black. 
    if(iterations >= maxIterations){
        brightness = 0;
        SDL_SetRenderDrawColor(renderer, brightness, brightness, brightness, 255);
    } else{
        SDL_SetRenderDrawColor(renderer, 255 - brightness, 255 - brightness, 255 - brightness, 255);
    }

    
    SDL_RenderDrawPoint(renderer, i, j);
}

/**
 * For one pixel, choose the brightness of that pixel based on the mandelbrot set's formula
 * @param renderer, pointer to the renderer of SDL which is where the pixel will be drawn (within the function)
 * @param i, the x coordinate of the pixel (within the resolution of the window–WINDOW_WIDTH and WINDOW_HEIGHT)
 * @param j, the y coordinate of the pixel (within the resolution of the window–WINDOW_WIDTH and WINDOW_HEIGHT)
 * @param minsize, scaling factor of the mandelbrot–based on how zoomed in the current image is or not
 * @param maxsize, scaling factor of the mandelbrot–based on how zoomed in the current image is or not
 * 
 * Quick explanation of scaling: The wider apart @param minsize and @param maxsize are, the more zoomed out
 * the image will be. If minsize + maxsize = 0, the image will be centered. If it's negative, the image will
 * be shifted right, and if positive, shifted left. 
 * 
 * Note: this is ONE pixel in one image of the mandelbrot–i.e. this function will be called WINDOW_WIDTH * WINDOW_HEIGHT
 * times.
 * 
 * This function draws the pixels according to the mandelbrot formula: z^2 + c;
 * The product of z^2 + c is applied into z, so the next iteration of z^2 + c = (z^2 + c)^2 + c
 * This is on the complex plane, so the values iterate and sometimes reach infinity, and sometimes don't. 
 * That is what the  MAX_ITERATIONS macro means–how many times we iterate this fomrula until it's too much. 
 * Technically, this can go on infinitely. But we cap it at 50, other tutorials cap this at 16 (the coding challenge)
 * Depending on the value and graphing this on the complex plane, coloring the values that result in a cycling towards infinity one color,
 * and coloring the other pixels white, we get the mandelbrot set fractal. The Julia set has a very similar iterative process, which
 * can have values that can be adjusted. 
 * 
 * This can be easily parallelized. For each zoom in, a new mandelbrot image is generated. 
 * This version of the function is based on the heatmap function–@KEVIN INSERT SOURCE HERE
 * that bases the color on the function heatmap..
 */
void choose_heatmap_mandelbrot(SDL_Renderer * renderer, int i, int j){
    double a = scale_number((double)i, 0, WINDOW_WIDTH, minimumReal, maximumReal);
    double b = scale_number((double)j, 0, WINDOW_HEIGHT, minimumComplex, maximumComplex);


    int iterations = 0; 
    int isInfinite = 16;
    double ca = a; // constant
    double cb = b; //constant

    //for each pixel, check if it's infinite or not
    while (iterations < maxIterations){
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
    // int brightness = scale_number((double)iterations, 0, MAXITERATIONS, 0, 255);
    
    int red = 0;
    int green = 0;
    int blue = 0;
    if (iterations < maxIterations) {
        heatmap(0, maxIterations, iterations, &red, &green, &blue);
    } else {
        red = 0;
        green = 0;
        blue = 0;
    }

    // Choose white color
    SDL_SetRenderDrawColor(renderer, red, green, blue, 255);
    //double scaled = scale_number((double)iterations, 0, maxIterations, 0, 1);
    //int brightness = scale_number(sqrt(scaled), 0, 1, 0, 255);

    // //if in mandelbrot make color black. 
    // if(iterations >= maxIterations){
    //     brightness = 0;
    // }
    // SDL_RenderDrawPoint(renderer, i, j);
}//end choose_heat_map_mandelbrot

/**
 * convert hsbtorbg
 */
SDL_Color hsbToRgb(float hue, float saturation, float brightness){
    SDL_Color rgb;
    float c, x, m;
    int h_i;

    // Normalize hsb values
    hue = fmod(hue, 360.0f);
    saturation = fmax(0.0f, fmin(saturation, 1.0f));
    brightness = fmax(0.0f, fmin(brightness, 1.0f));

    c = brightness * saturation;
    x = c * (1.0f - fabs(fmod(hue / 60.0f, 2.0f) - 1.0f));
    m = brightness - c;

    h_i = (int)(hue / 60.0f);

    switch (h_i) {
        case 0:
            rgb.r = c + m; rgb.g = x + m; rgb.b = m; break;
        case 1:
            rgb.r = x + m; rgb.g = c + m; rgb.b = m; break;
        case 2:
            rgb.r = m; rgb.g = c + m; rgb.b = x + m; break;
        case 3:
            rgb.r = m; rgb.g = x + m; rgb.b = c + m; break;
        case 4:
            rgb.r = x + m; rgb.g = m; rgb.b = c + m; break;
        case 5:
            rgb.r = c + m; rgb.g = m; rgb.b = x + m; break;
        default:
            rgb.r = 0; rgb.g = 0; rgb.b = 0; break;
    }

    return rgb;
}

/**
 * For one pixel, choose the brightness of that pixel based on the mandelbrot set's formula
 * @param renderer, pointer to the renderer of SDL which is where the pixel will be drawn (within the function)
 * @param i, the x coordinate of the pixel (within the resolution of the window–WINDOW_WIDTH and WINDOW_HEIGHT)
 * @param j, the y coordinate of the pixel (within the resolution of the window–WINDOW_WIDTH and WINDOW_HEIGHT)
 * @param minsize, scaling factor of the mandelbrot–based on how zoomed in the current image is or not
 * @param maxsize, scaling factor of the mandelbrot–based on how zoomed in the current image is or not
 * 
 * Quick explanation of scaling: The wider apart @param minsize and @param maxsize are, the more zoomed out
 * the image will be. If minsize + maxsize = 0, the image will be centered. If it's negative, the image will
 * be shifted right, and if positive, shifted left. 
 * 
 * Note: this is ONE pixel in one image of the mandelbrot–i.e. this function will be called WINDOW_WIDTH * WINDOW_HEIGHT
 * times.
 * 
 * This function draws the pixels according to the mandelbrot formula: z^2 + c;
 * The product of z^2 + c is applied into z, so the next iteration of z^2 + c = (z^2 + c)^2 + c
 * This is on the complex plane, so the values iterate and sometimes reach infinity, and sometimes don't. 
 * That is what the  MAX_ITERATIONS macro means–how many times we iterate this fomrula until it's too much. 
 * Technically, this can go on infinitely. But we cap it at 50, other tutorials cap this at 16 (the coding challenge)
 * Depending on the value and graphing this on the complex plane, coloring the values that result in a cycling towards infinity one color,
 * and coloring the other pixels white, we get the mandelbrot set fractal. The Julia set has a very similar iterative process, which
 * can have values that can be adjusted. 
 * 
 * This can be easily parallelized. For each zoom in, a new mandelbrot image is generated. 
 * This version of the function is colored
 */
void choose_colorful_mandelbrot(SDL_Renderer * renderer, int i, int j){
    double a = scale_number((double)i, 0, WINDOW_WIDTH, minimumReal, maximumReal); //real
    double b = scale_number((double)j, 0, WINDOW_HEIGHT, minimumComplex, maximumComplex); //complex

    int iterations = 0; 
    int isInfinite = 25;
    double ca = a; // constant
    double cb = b; //constant
    double check_bounds;

    //for each pixel, check if it's infinite or not
    while (iterations < maxIterations){
        double aa = a * a - b * b;
        double bb = 2 * a * b;

        a = aa + ca;
        b = bb + cb;

        check_bounds = a + b;
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

    /**
     * bounds, infinite = if our bounds are 
     * iterations max iterations = if the iterations hits max iterations, we exit early 
     */

    SDL_Color myColor;
    double scaled_bounds = scale_number((double)check_bounds, 0, isInfinite, 0, 1);
    double scaled_it = scale_number((double)iterations, 0, maxIterations, 0, 1);
    myColor.b = scale_number(scaled_it,0, 1, 0, 255);
    int brightness = scale_number(sqrt(scaled_bounds), 0, 1, 0, 255);

    if(check_bounds < isInfinite){
        myColor.r = 0;
        myColor.b = 0;
        myColor.g = 0;
    }
    

    SDL_SetRenderDrawColor(renderer, myColor.r, myColor.b, myColor.g, brightness);
    SDL_RenderDrawPoint(renderer, i, j);
}


void choose_colorful_julia(SDL_Renderer * renderer,int i, int j){
     double a = scale_number((double)i, 0, WINDOW_WIDTH, minimumReal, maximumReal); //real
    double b = scale_number((double)j, 0, WINDOW_HEIGHT, minimumComplex, maximumComplex); //complex

    int iterations = 0; 
    int isInfinite = 25;
    double ca = a; // constant
    double cb = b; //constant

    //for each pixel, check if it's infinite or not
    while (iterations < maxIterations){
        double aa = a * a - b * b;
        double bb = 2 * a * b;

        a = aa + constantA;
        b = bb + constantB;

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
    double scaled = scale_number((double)iterations, 0, maxIterations, 0, 1);
    int brightness = scale_number(sqrt(scaled), 0, 1, 0, 255);

    //if in mandelbrot make color white. 
    if(iterations == maxIterations){
        brightness = 255;
    }

    SDL_SetRenderDrawColor(renderer, brightness, brightness, 255, 255);
    SDL_RenderDrawPoint(renderer, i, j);
}

/**
 * Iteratively calls @function choose_brightness_mandelbrot or @function choose_heatmap_mandelbrot based on
 * the choice 
 * @param choice – decides which fractal we will use
 * @param renderer – the renderer in SDL which is where the mandelbrot image will be drawn
 * @param scale – the scale parameter which is adjusted in MAIN based on the SDL scaling, will be updated as it is passed into this function
 */
void create_mandelbrot(SDL_Renderer * renderer, int choice){
    double halfWidth = baseWidth*zoomScale/2.0;
    double halfHeight = baseHeight*zoomScale/2.0;
    
    minimumReal = centerX - halfWidth; //distance to max = zoomin/out;
    maximumReal = centerX + halfWidth; //distance to min = zoomin/out; 
    minimumComplex = centerY + halfHeight; 
    maximumComplex = centerY -  halfHeight;
    
    for(int i = 0; i < WINDOW_WIDTH; i++) {
        for(int j = 0; j < WINDOW_HEIGHT; j++) {
            //the distance between minsize and maxsize = the zooming scale
            //if it's right centered or left centered around 0 = which part of mandelbrot

            /**
             * base height/width. The distance between is how zoomed in
             * centerX is the overall shift left or right (between minsize and maxsize)
             * centerY is the 
             * 
             * zoomScale is the distance between minsize and maxsize; greater the value, the greater the distance
             * 
             */

    

            //choose which version of the mandelbrot to use
            switch (choice){
                case 1:
                    choose_brightness_mandelbrot(renderer, i, j);
                    break;
                case 2:
                    choose_colorful_mandelbrot(renderer, i, j);
                case 3:
                    choose_heatmap_mandelbrot(renderer, i, j);
                case 4: 
                    choose_colorful_julia(renderer, i, j);
                // default:
                //     choose_brightness_mandelbrot(renderer, i, j);
                //     printf("which MAN: %d\n", 1);
                //     break;
            }
            
        }//end for
    }//end for

}//end create_Mandelbrot


/**
 * max – returns the highest value between a and b
 * @param a the first integer input
 * @param b second integer input
 */
int max(int a, int b){
    return a > b ? a : b;
} 

// This function takes a value and maps it to a color on a heatmap according
void heatmap(float minimum, float maximum, float val, int* r, int* g, int* b) {
    float ratio = 2 * (val - minimum) / (maximum - minimum);
    *b = max(0, (int) 255*(1 - ratio));
    *r = max(0, (int) 255*(ratio - 1));
    *g = 255 - *b - *r;
}

/**
 * Scalenumber–maps a current value index of a one-dimensional field onto a larger field.
 * eg: 9.5 on a 0-10 scale would be 95 on a 0-100 scale; the inputs would be scale_number(9.5, 0, 10, 0, 100);
 * 
 * @param cv the current value that we want to transform
 * @param min the current scale (one-dimensional) minimum
 * @param max the current scale (one-dimensional) maximum
 * @param nmin the transformed scale (one-dimensional) minimum
 * @param nmax the transformed scale (one-dimensional) maximum
 */
double scale_number(double cv, double min, double max, double nmin, double nmax){
    double range1 = max - min;
    double range2 = nmax - nmin;

    double new_number = ((cv * range2)/range1) + (nmin);
    return new_number;
}

/*
 * taken from the web
 
void save_texture(SDL_Renderer *ren, SDL_Texture *tex, const char *filename){
    SDL_Texture *ren_tex;
    SDL_Surface *surf;
    int st;
    int w;
    int h;
    int format;
    void *pixels;

    pixels  = NULL;
    surf    = NULL;
    ren_tex = NULL;
    format  = SDL_PIXELFORMAT_RGBA32;

    // Get information about texture we want to save 
    st = SDL_QueryTexture(tex, NULL, NULL, &w, &h);
    if (st != 0) {
        SDL_Log("Failed querying texture: %s\n", SDL_GetError());
        goto cleanup;
    }

    ren_tex = SDL_CreateTexture(ren, format, SDL_TEXTUREACCESS_TARGET, w, h);
    if (!ren_tex) {
        SDL_Log("Failed creating render texture: %s\n", SDL_GetError());
        goto cleanup;
    }

    //
    // Initialize our canvas, then copy texture to a target whose pixel data we 
    // can access
    //
    st = SDL_SetRenderTarget(ren, ren_tex);
    if (st != 0) {
        SDL_Log("Failed setting render target: %s\n", SDL_GetError());
        goto cleanup;
    }

    SDL_SetRenderDrawColor(ren, 0x00, 0x00, 0x00, 0x00);
    SDL_RenderClear(ren);

    st = SDL_RenderCopy(ren, tex, NULL, NULL);
    if (st != 0) {
        SDL_Log("Failed copying texture data: %s\n", SDL_GetError());
        goto cleanup;
    }

    // Create buffer to hold texture data and load it 
    pixels = malloc(w * h * SDL_BYTESPERPIXEL(format));
    if (!pixels) {
        SDL_Log("Failed allocating memory\n");
        goto cleanup;
    }

    st = SDL_RenderReadPixels(ren, NULL, format, pixels, w * SDL_BYTESPERPIXEL(format));
    if (st != 0) {
        SDL_Log("Failed reading pixel data: %s\n", SDL_GetError());
        goto cleanup;
    }

    // Copy pixel data over to surface 
    surf = SDL_CreateRGBSurfaceWithFormatFrom(pixels, w, h, SDL_BITSPERPIXEL(format), w * SDL_BYTESPERPIXEL(format), format);
    if (!surf) {
        SDL_Log("Failed creating new surface: %s\n", SDL_GetError());
        goto cleanup;
    }

    // Save result to an image 
    st = SDL_SaveBMP(surf, filename);
    if (st != 0) {
        SDL_Log("Failed saving image: %s\n", SDL_GetError());
        goto cleanup;
    }

    SDL_Log("Saved texture as BMP to \"%s\"\n", filename);

cleanup:
    SDL_FreeSurface(surf);
    free(pixels);
    SDL_DestroyTexture(ren_tex);
}
*/


/**
 * Notes from Ellie
 * Get vectors instead of pixels
 * pixels are rastor data – assigning points on a screen
 * 
 * Vectorized data – a different kind of line that comes out of adobe
 * - when scaled up, it looks just as high definition
 * - it scales distances rather than specific points. 
 */