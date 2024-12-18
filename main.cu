#include <SDL.h> // include SDL<-- this must be linked properly in whatever programming is made
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <pthread.h>

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
#define WINDOW_WIDTH 1080
#define WINDOW_HEIGHT 1080

// Number of threads per block, should be a multiple of 32
#define THREADSPERBLOCK 256


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
double baseWidth = 4.0;  // Base width of the view in complex plane
double baseHeight = 4.0;  // Base width of the view in complex plane
double keyboardSpeed = 0.1;  // Speed of scrolling
int which = 1; //which set are we using?
double minimumReal;
double maximumReal;
double maximumComplex;
double minimumComplex;
double constantA = 0.0;  // Julia set real constant
double constantB = 0.0;  // Julia set imaginary constant
double isInfinite = 25;

double juliaAnimationSpeed = 0.01;  // Controls how fast the constants change
bool autoAnimate = false;  // Toggle for auto-animation
double animationTime = 0.0;  // Tracks time for smooth animation

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
void place_bright_mandelbrot_pixel(SDL_Renderer * renderer, int i, int j);


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
void place_heatmap_mandelbrot_pixel(SDL_Renderer * renderer, int i, int j);

/**
 * Iteratively calls @function choose_brightness_mandelbrot or @function choose_heatmap_mandelbrot based on
 * the choice 
 * @param which  decides which fractal we will use
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
                        maxIterations *= 1.01;
                        break;
                    case SDLK_o:
                        maxIterations /= 1.01;
                        break;
                    case SDLK_g:
                        isInfinite++;
                    case SDLK_h:
                        isInfinite--;
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
                    /*Adjust parameters on julia set*/
                    case SDLK_j:
                        constantA += 0.01;
                        printf("Julia constant A: %.4f\n", constantA);  // Added feedback
                        break;
                    case SDLK_k:
                        constantA -= 0.01;
                        printf("Julia constant A: %.4f\n", constantA);  // Added feedback
                        break;
                    case SDLK_s:
                        constantB += 0.01;
                        printf("Julia constant B: %.4f\n", constantB);  // Added feedback
                        break;
                    case SDLK_d:
                        constantB -= 0.01;
                        printf("Julia constant B: %.4f\n", constantB);  // Added feedback
                        break;

                    /* F for preset julia set*/
                    case SDLK_f:
                        constantA = -0.835;
                        constantB = -0.2321;
                        printf("Julia preset: A=%.4f, B=%.4f\n", constantA, constantB);  // Added feedback
                        break;
                    case SDLK_r:
                        // Reset to original view
                        centerX = 0.0;
                        centerY = 0.0;
                        zoomScale = 1.0;
                        maxIterations = 100;
                        break;
                    case SDLK_SPACE:  // Space bar to toggle animation
                        autoAnimate = !autoAnimate;
                        printf("Animation %s\n", autoAnimate ? "started" : "stopped");
                        break;
                    case SDLK_m:  // 'M' key to modify animation speed
                        juliaAnimationSpeed *= 1.5;
                        printf("Animation speed: %.4f\n", juliaAnimationSpeed);
                        break;
                    case SDLK_n:  // 'N' key to decrease animation speed
                        juliaAnimationSpeed /= 1.5;
                        printf("Animation speed: %.4f\n", juliaAnimationSpeed);
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
void place_bright_mandelbrot_pixel(SDL_Renderer * renderer, int i, int j){
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
    }

    SDL_SetRenderDrawColor(renderer, brightness, brightness, brightness, 255);
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
void place_heatmap_mandelbrot_pixel(SDL_Renderer * renderer, int i, int j){
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
    double scaled = scale_number((double)iterations, 0, maxIterations, 0, 1);
    int brightness = scale_number(sqrt(scaled), 0, 1, 0, 255);

    //if in mandelbrot make color black. 
    if(iterations >= maxIterations){
        brightness = 0;
    }
    SDL_RenderDrawPoint(renderer, i, j);
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
    // this is a variable I haven't experienced with much. 
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
    double scaled = scale_number((double)iterations, 0, maxIterations, 0, 1);

    /**
     * Parameters to mess with:
     * iterations/maxiterations
     * ca (constant portion)
     * cb (constant portion)
     */
    int brightness = scale_number(sqrt(scaled), 0, 1, 0, 255);

    

    //if in mandelbrot make color black. 
    if(iterations >= maxIterations){
        brightness = 0;
    }

    double color = sqrt(iterations/maxIterations);
    if(iterations == maxIterations){
        color = 0;
    }
    SDL_SetRenderDrawColor(renderer, color, color, color, color);
    SDL_RenderDrawPoint(renderer, i, j);
}

__device__ double scale_number_device(double cv, double min, double max, double nmin, double nmax){
    double range1 = max - min;
    double range2 = nmax - nmin;

    double new_number = ((cv * range2)/range1) + (nmin);
    return new_number;
}

__global__ void brightness_mandelbrot(
    int windowHeight, 
    int windowWidth, 
    double baseWidth, 
    double baseHeight, 
    double centerX, 
    double centerY, 
    double zoomScale, 
    int maxIter, 
    int isInfinite, 
    double* pixel_double) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int j = index / windowWidth;
    int i = index % windowWidth;
    
    // Check if this thread is within bounds
    if (i >= windowWidth || j >= windowHeight) {
        return;
    }
    
    // get the complex number at (i, j)
    double halfWidth = baseWidth*zoomScale/2.0;
    double halfHeight = baseHeight*zoomScale/2.0;
    /**
     * base height/width. The distance between is how zoomed in
     * centerX is the overall shift left or right (between minsize and maxsize)
     * centerY is the overall shift up or down
     * zoomScale is the distance between minsize and maxsize; greater the value, the greater the distance
     */
    double minimumReal = centerX - halfWidth;
    double maximumReal = centerX + halfWidth;
    double minimumComplex = centerY + halfHeight; 
    double maximumComplex = centerY -  halfHeight;

    double a = scale_number_device((double)i, 0, windowWidth, minimumReal, maximumReal); //real
    double b = scale_number_device((double)j, 0, windowHeight, minimumComplex, maximumComplex); //complex

    // C = Z_0
    double ca = a; // constant
    double cb = b; //constant

    // other variables
    int iterations = 0; 
    double check_bounds;

    // while loop to check if the pixel at index is infinite or not
    while (iterations < maxIter){
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

    // save the double to pixel_double
    pixel_double[index] = ((double) iterations)/maxIter;
}

__global__ void brightness_julia(
    int windowHeight, 
    int windowWidth, 
    double baseWidth, 
    double baseHeight, 
    double centerX, 
    double centerY, 
    double zoomScale,
    double constantA,  // Julia set parameter
    double constantB,  // Julia set parameter 
    int maxIter, 
    int isInfinite, 
    double* pixel_double) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int j = index / windowWidth;
    int i = index % windowWidth;
    
    // Check if this thread is within bounds
    if (i >= windowWidth || j >= windowHeight) {
        return;
    }
    
    // Calculate bounds
    double halfWidth = baseWidth*zoomScale/2.0;
    double halfHeight = baseHeight*zoomScale/2.0;
    double minimumReal = centerX - halfWidth;
    double maximumReal = centerX + halfWidth;
    double minimumComplex = centerY + halfHeight; 
    double maximumComplex = centerY - halfHeight;

    // Get initial complex number coordinates
    double a = scale_number_device((double)i, 0, windowWidth, minimumReal, maximumReal);
    double b = scale_number_device((double)j, 0, windowHeight, minimumComplex, maximumComplex);

    int iterations = 0;
    double check_bounds;

    // Julia set iteration
    while (iterations < maxIter) {
        double aa = a * a - b * b;
        double bb = 2 * a * b;

        // Key difference from Mandelbrot: we use constant values instead of original coordinates
        a = aa + constantA;
        b = bb + constantB;

        check_bounds = a + b;
        if (check_bounds < 0) {
            check_bounds *= -1;
        }

        if (check_bounds > isInfinite) {
            break;
        }
        iterations++;
    }

    pixel_double[index] = ((double)iterations) / maxIter;
}

// Structure to pass data to threads
typedef struct {
    double* pixel_double;
    Uint32* pixels;
    int start_row;
    int end_row;
    int width;
} ThreadData;

// Thread function to process pixels
void* process_pixels(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    
    for(int j = data->start_row; j < data->end_row; j++) {
        for(int i = 0; i < data->width; i++) {
            Uint8 brightness = (Uint8)(data->pixel_double[j * data->width + i] * 255);
            // save the RGBA pixel value for SDL rendering
            data->pixels[j * data->width + i] = (brightness << 24) |  // Red byte
                                                (brightness << 16) |  // Green byte
                                                (brightness << 8) | // Blue byte
                                                (0xFF); // opacity byte (always full opacity)
        }
    }
    
    return NULL;
}

/**
 * Iteratively calls @function choose_brightness_mandelbrot or @function choose_heatmap_mandelbrot based on
 * the choice 
 * @param choice – decides which fractal we will use
 * @param renderer – the renderer in SDL which is where the mandelbrot image will be drawn
 * @param scale – the scale parameter which is adjusted in MAIN based on the SDL scaling, will be updated as it is passed into this function
 */
void create_mandelbrot(SDL_Renderer* renderer, int choice) {
    double halfWidth = baseWidth*zoomScale/2.0;
    double halfHeight = baseHeight*zoomScale/2.0;

        /**
     * base height/width. The distance between is how zoomed in
     * centerX is the overall shift left or right (between minsize and maxsize)
     * centerY is the 
     * 
     * zoomScale is the distance between minsize and maxsize; greater the value, the greater the distance
     * 
     */
    minimumReal = centerX - halfWidth; //distance to max = zoomin/out;
    maximumReal = centerX + halfWidth; //distance to min = zoomin/out; 
    minimumComplex = centerY + halfHeight; 
    maximumComplex = centerY -  halfHeight;

    // A 1-D arrray of pixels of values 0~1 where pixel ij is j * WINDOW_WIDTH + i
    double* cpu_pixel_double = (double*)malloc(sizeof(double) * WINDOW_HEIGHT * WINDOW_WIDTH);
    double* gpu_pixel_double;
    if(cudaMalloc(&gpu_pixel_double, sizeof(double) * WINDOW_HEIGHT * WINDOW_WIDTH) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate pixel_double array on GPU\n");
        exit(2);
    }

    // we don't need to copy values to gpu because there is no value to be copied
    // gpu will compute the double values and we will pass it back to cpu_pixel_double afterwards. 


    // Calculate total number of pixels
    int totalPixels = WINDOW_HEIGHT * WINDOW_WIDTH;
    
    // Calculate grid dimensions to ensure we cover all pixels
    dim3 threadsPerBlock(THREADSPERBLOCK);
    dim3 numBlocks((totalPixels + THREADSPERBLOCK - 1) / THREADSPERBLOCK);

    // Choose which kernel to run based on choice
    if (choice == 1) {  // Mandelbrot variations
        brightness_mandelbrot<<<numBlocks, threadsPerBlock>>>(
            WINDOW_HEIGHT, WINDOW_WIDTH, baseWidth, baseHeight, 
            centerX, centerY, zoomScale, maxIterations, isInfinite, 
            gpu_pixel_double);
    } else {  // Julia set
        brightness_julia<<<numBlocks, threadsPerBlock>>>(
            WINDOW_HEIGHT, WINDOW_WIDTH, baseWidth, baseHeight, 
            centerX, centerY, zoomScale, constantA, constantB,
            maxIterations, isInfinite, gpu_pixel_double);
    }

    // transfer gpu_pixel_double to cpu_pixel_double
    cudaMemcpy(cpu_pixel_double, gpu_pixel_double, sizeof(double) * WINDOW_HEIGHT * WINDOW_WIDTH, cudaMemcpyDeviceToHost);

    // After getting cpu_pixel_double from CUDA:
    
    // Create pixel buffer for SDL texture
    Uint32* pixels = (Uint32*)malloc(WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(Uint32));
    
    // Create threads
    const int num_threads = 20;  
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    
    // Calculate rows per thread
    int rows_per_thread = WINDOW_HEIGHT / num_threads;
    
    // Create and launch threads
    for(int i = 0; i < num_threads; i++) {
        thread_data[i].pixel_double = cpu_pixel_double;
        thread_data[i].pixels = pixels;
        thread_data[i].width = WINDOW_WIDTH;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == num_threads - 1) ? WINDOW_HEIGHT : (i + 1) * rows_per_thread;
        
        if(pthread_create(&threads[i], NULL, process_pixels, &thread_data[i])) {
            fprintf(stderr, "Error creating thread\n");
            exit(1);
        }
    }
    
    // Wait for all threads to complete
    for(int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Create and update texture
    SDL_Texture* texture = SDL_CreateTexture(renderer,
        SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_STREAMING,
        WINDOW_WIDTH, WINDOW_HEIGHT);
    
    SDL_UpdateTexture(texture, NULL, pixels, WINDOW_WIDTH * sizeof(Uint32));
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    
    // Cleanup
    SDL_DestroyTexture(texture);
    free(pixels);

    // free gpu_pixel_double
    cudaFree(gpu_pixel_double);

    // free cpu_pixel_double
    free(cpu_pixel_double);
    
    // for(int i = 0; i < WINDOW_WIDTH; i++) {
    //     for(int j = 0; j < WINDOW_HEIGHT; j++) {
    //         //the distance between minsize and maxsize = the zooming scale
    //         //if it's right centered or left centered around 0 = which part of mandelbrot

    //         //choose which version of the mandelbrot to use
    //         switch (choice)
    //         {
    //         case 1:
    //             choose_brightness_mandelbrot(renderer, i, j);
    //             break;
    //         case 2:
    //             choose_colorful_mandelbrot(renderer, i, j);
    //             break;
    //         case 3:
    //             choose_heatmap_mandelbrot(renderer, i, j);
    //         // default:
    //         //     choose_brightness_mandelbrot(renderer, i, j);
    //         //     printf("which MAN: %d\n", 1);
    //         //     break;
    //         }
            
    //     }//end for
    // }//end for

}//end create_Mandelbrot


/**
 * max – returns the highest value between a and b
 * @param a the first integer input
 * @param b second integer input
 */
// int max(int a, int b){
//     return a > b ? a : b;
// } 

// This function takes a value and maps it to a color on a heatmap according
void heatmap(float minimum, float maximum, float val, int* r, int* g, int* b) {
    float ratio = 2 * (val - minimum) / (maximum - minimum);
    *b = max(0.0, (int) 255*(1 - ratio));
    *r = max(0.0, (int) 255*(ratio - 1));
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

/**
 * Notes from Ellie
 * Get vectors instead of pixels
 * pixels are rastor data – assigning points on a screen
 * 
 * Vectorized data – a different kind of line that comes out of adobe
 * - when scaled up, it looks just as high definition
 * - it scales distances rather than specific points. 
 */