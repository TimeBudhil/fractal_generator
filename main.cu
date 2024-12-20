#include <SDL.h> // include SDL<-- this must be linked properly in whatever programming is made
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <pthread.h>

// This project is a GPU-accelerated fractal generator using CUDA.
// We (Rhys, Budhil, and Kevin) took significant inspiration for the serial implementation of generating Mandelbrot set and Julia sets from these videos by The Coding Train:
// https://www.youtube.com/watch?v=fAsaSkmbF5s&t=12s
// https://www.youtube.com/watch?v=6z7GQewK-Ks&t=967s

// Global variables for the window size
// The smaller the window, the more easily the pages will load. 
#define WINDOW_WIDTH 1080
#define WINDOW_HEIGHT 1080

// Number of threads per block, should be a multiple of 32
#define THREADSPERBLOCK 256


// Global variable for the number of iterations to be run per individual pixel
// If you zoom in increase the cap of these iterations
// Zoomed in coordinate conversion just another coordinate conversion
int maxIterations = 100;  // the higher this value is the, more "concrete" the fractal will look

// Initialize the variables to click the screen to move and zoom in or out.
int mouseClickX = -1;
int mouseClickY = -1;


// Global variables for modifying specifications and parameters of the fractals
double centerX = 0.0;  // how much right or left we go.  (multiply against zoomscale to be proportional) on complex plane
double centerY = 0.0;  //how much up or down we go (multiply against zoomscale to be proportional) on complex plane
double zoomScale = 1.0;  // Zoom scale factor
double baseWidth = 4.0;  // Base width of the view in complex plane
double baseHeight = 4.0;  // Base width of the view in complex plane
double keyboardSpeed = 0.1;  // Speed of scrolling
int which = 1; //which set we are using
double minimumReal; // minimum real value
double maximumReal; // maximum real value
double maximumComplex; // maximum complex value
double minimumComplex; // minimum complex value
double constantA = 0.0;  // Julia set real constant
double constantB = 0.0;  // Julia set imaginary constant
double isInfinite = 25; // The threashold when the value diverges to infinity

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
 * Calls kernel brightness_mandelbrot or kernel brightness_julia based on the value of which
 * 
 * @param which  decides which fractal we will use
 * @param renderer – the renderer in SDL which is where the mandelbrot image will be drawn
 */
void create_fractal(SDL_Renderer * renderer, int which);

// main function
int main(int argc, char* argv[]) {

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

    /* Main functions */
    /*
     * Function guide for understanding SDL:
     * SDL_SetRenderDrawColor changes the color of the pixel to be drawn
     * SDL_RenderDrawPoint draws the color
     */

    // Clear screen (black background) in the beginning
    //clearing the renderer
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer); 

    // MAIN loop the event handling and creation of the mandelbrot 
    // Initialize boolean while loop, for now we aren't quitting, so to false
    bool quit = false;
    while (!quit) {

        //handle events using the SDL event systems
        /**CONTROLS:
         * 1. zooming in & zooming out: zooming with scroller; moving right left up down based on right left up down keys
         * 2. moving right and left:
         * 3. changing fractal image: 1, 2 to choose between:
         *      1–mandelbrot set
         *      2–Julia set
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

                    // get the mouse position
                    int x, y;
                    SDL_GetMouseState(&x, &y);
                    printf("Mouse clicked at (%d, %d)\n", x, y);

                    // Calculate the bounds of the current view
                    double halfWidth = baseWidth * zoomScale / 2.0;
                    double halfHeight = baseHeight * zoomScale / 2.0;
                    minimumReal = centerX - halfWidth;  
                    maximumReal = centerX + halfWidth;
                    minimumComplex = centerY + halfHeight;
                    maximumComplex = centerY - halfHeight;

                    // Convert mouse coordinates to complex plane coordinates
                    double mouseX = scale_number(x, 0, WINDOW_WIDTH, minimumReal, maximumReal);
                    double mouseY = scale_number(y, 0, WINDOW_HEIGHT, minimumComplex, maximumComplex);
                    
                    // Calculate the offset from current center to mouse position
                    double dx = mouseX - centerX;
                    double dy = mouseY - centerY;

                    // Apply zoom
                    double zoomFactor = 0.9;
                    zoomScale *= zoomFactor;
                    
                    // Adjust center position to keep mouse point invariant
                    centerX = mouseX - dx * zoomFactor;
                    centerY = mouseY - dy * zoomFactor;
                }//end if

                /*Zooming in and Out*/
            } else if (event.type == SDL_KEYDOWN) {
                /**Ideally, the scrolling would move the screen in and out
                 * secondly, a scroll right or left would move the screen right or left
                 */

                // moving right and left based on keyboard typing 
                // Keys are: left, right, up, down, r to reset, and esc to secape
                switch (event.key.keysym.sym) {
                    // escape quits everything
                    case SDLK_ESCAPE:
                        quit = true;
                        break;
                    case SDLK_LEFT:
                        // Direction key left (move view to the left)
                        centerX -= keyboardSpeed * zoomScale;
                        break;
                    case SDLK_RIGHT:
                        // Direction key right (move view to the right)
                        centerX += keyboardSpeed * zoomScale;
                        break;
                    case SDLK_UP:
                        // Direction key up (move view up)
                        centerY += keyboardSpeed * zoomScale;
                        break;
                    case SDLK_DOWN:
                        // Direction key down (move view down)
                        centerY -= keyboardSpeed * zoomScale;
                        break;
                    case SDLK_EQUALS:
                        // Zoom in
                        zoomScale *= 0.9;
                        break;
                    case SDLK_MINUS:
                        // Zoom out
                        zoomScale /= 0.9;
                        break;
                    case SDLK_i:
                        // Increase the number of iterations
                        maxIterations *= 1.01;
                        break;
                    case SDLK_o:
                        // Decrease the number of iterations
                        maxIterations /= 1.01;
                        break;
                    case SDLK_n: //n for negative
                        // Flip the zoom scale
                        zoomScale *= -1;
                        break;
                    case SDLK_g:
                        // Increase the threshold for divergence
                        isInfinite++;
                    case SDLK_h:
                        // Decrease the threshold for divergence
                        isInfinite--;
                    case SDLK_1: 
                        which = 1;
                        printf("which: %d\n", which);
                        break;
                    case SDLK_2: 
                        which = 2;
                        printf("which: %d\n", which);
                        break;
                    case SDLK_j:
                        // Increase the constant A for the Julia set
                        constantA += 0.01;
                        printf("Julia constant A: %.4f\n", constantA);  // Added feedback
                        break;
                    case SDLK_k:
                        // Decrease the constant A for the Julia set
                        constantA -= 0.01;
                        printf("Julia constant A: %.4f\n", constantA);  // Added feedback
                        break;
                    case SDLK_s:
                        // Increase the constant B for the Julia set
                        constantB += 0.01;
                        printf("Julia constant B: %.4f\n", constantB);  // Added feedback
                        break;
                    case SDLK_d:
                        // Decrease the constant B for the Julia set
                        constantB -= 0.01;
                        printf("Julia constant B: %.4f\n", constantB);  // Added feedback
                        break;

                    // F for a julia set with a preset constant A and B
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
                }

                /**Scrolling based on teh scale */
            } else if (event.type == SDL_MOUSEWHEEL) {
                // Zoom logic (same as before)
                if (event.wheel.y > 0) { // Scroll up (zoom in)
                    zoomScale *= 0.9;  // Zoom in by reducing scale
                } else if (event.wheel.y < 0) { // Scroll down (zoom out)
                    zoomScale /= 0.9;  // Zoom out by increasing scale
                } 
                
            }
        }

        // Clear screen
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // create the mandelbrot
        create_fractal(renderer, which);

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
 * @brief Scales a number from one range to another on GPU
 * 
 * @param cv the current value that we want to transform
 * @param min the current scale (one-dimensional) minimum
 * @param max the current scale (one-dimensional) maximum
 * @param nmin the transformed scale (one-dimensional) minimum
 * @param nmax the transformed scale (one-dimensional) maximum
 * @return the new number
 */
__device__ double scale_number_device(double cv, double min, double max, double nmin, double nmax){
    double range1 = max - min;
    double range2 = nmax - nmin;

    double new_number = ((cv * range2)/range1) + (nmin);
    return new_number;
}

/**
 * @brief GPU kernel for generating a Mandelbrot set using brightness values
 * 
 * @param windowHeight the height of the window
 * @param windowWidth the width of the window
 * @param baseWidth the base width of the fractal
 * @param baseHeight the base height of the fractal
 * @param centerX the center x coordinate of the fractal
 * @param centerY the center y coordinate of the fractal
 * @param zoomScale the zoom scale of the fractal
 * @param maxIter the maximum number of iterations
 * @param isInfinite the threshold for divergence
 * @param pixel_values the array to store the pixel values
 */
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
    uint8_t* pixel_values) {

    // get the index of the pixel
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // get the row and column of the corresponding pixel
    int j = index / windowWidth;
    int i = index % windowWidth;
    
    // Check if this thread is within bounds
    if (i >= windowWidth || j >= windowHeight) {
        return;
    }
    
    // get the complex number at this pixel (i, j)
    double halfWidth = baseWidth*zoomScale/2.0;
    double halfHeight = baseHeight*zoomScale/2.0;

    // get the minimum and maximum real and complex values
    double minimumReal = centerX - halfWidth;
    double maximumReal = centerX + halfWidth;
    double minimumComplex = centerY + halfHeight; 
    double maximumComplex = centerY -  halfHeight;

    // get the real and complex values at this pixel
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
        // calculate the new a and b
        double aa = a * a - b * b; // real part
        double bb = 2 * a * b; // imaginary part

        // add the constant to the new a and b
        a = aa + ca;
        b = bb + cb;

        // get the absolute value of the a + b
        check_bounds = a + b;
        if(check_bounds < 0){
            check_bounds *= -1;
        }

        // check if it surpasses the threshold
        if(check_bounds > isInfinite){
            break;
        }

        // increment the iteration
        iterations++;
    }//end while

    // Calculate the brightness value as a uint8_t
    pixel_values[index] = (uint8_t)(((float)iterations / maxIter) * 255);
}

/**
 * @brief GPU kernel for generating a Julia set using brightness values
 * 
 * @param windowHeight the height of the window
 * @param windowWidth the width of the window
 * @param baseWidth the base width of the fractal
 * @param baseHeight the base height of the fractal
 * @param centerX the center x coordinate of the fractal
 * @param centerY the center y coordinate of the fractal
 * @param zoomScale the zoom scale of the fractal
 * @param constantA the constant A for the Julia set
 * @param constantB the constant B for the Julia set
 * @param maxIter the maximum number of iterations
 * @param isInfinite the threshold for divergence
 * @param pixel_values the array to store the pixel values
 */
__global__ void brightness_julia(
    int windowHeight, 
    int windowWidth, 
    double baseWidth, 
    double baseHeight, 
    double centerX, 
    double centerY, 
    double zoomScale,
    double constantA, 
    double constantB, 
    int maxIter, 
    int isInfinite, 
    uint8_t* pixel_values) {

    // get the index of the pixel
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // get the row and column of the corresponding pixel
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

    // get the initial complex number coordinates
    double a = scale_number_device((double)i, 0, windowWidth, minimumReal, maximumReal);
    double b = scale_number_device((double)j, 0, windowHeight, minimumComplex, maximumComplex);

    int iterations = 0;
    double check_bounds;

    // while loop to check if the pixel at index is infinite or not
    while (iterations < maxIter) {
        // calculate the new a and b
        double aa = a * a - b * b;
        double bb = 2 * a * b;

        // Key difference from Mandelbrot: we use constant values instead of original coordinates
        a = aa + constantA;
        b = bb + constantB;

        // get the absolute value of the a + b
        check_bounds = a + b;
        if (check_bounds < 0) {
            check_bounds *= -1;
        }

        // check if it surpasses the threshold
        if (check_bounds > isInfinite) {
            break;
        }

        // increment the iteration
        iterations++;
    }

    // calculate the brightness value as a uint8_t
    pixel_values[index] = (uint8_t)(((float)iterations / maxIter) * 255);
}

/**
 * @brief Structure to pass data to threads
 */
typedef struct {
    uint8_t* pixel_values;
    Uint32* pixels;
    int start_row;
    int end_row;
    int width;
} ThreadData;

/**
 * @brief Thread function to convert the brightness values between 0~1 to 4-byte RGBA values for SDL rendering
 * 
 * @param arg the argument to the thread
 * @return NULL
 */
void* process_pixels(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    
    // iterate through the rows and columns of the pixel
    for(int j = data->start_row; j < data->end_row; j++) {
        for(int i = 0; i < data->width; i++) {

            // make sure the j and i are within bounds
            if(j >= WINDOW_HEIGHT || i >= WINDOW_WIDTH) {
                continue;
            }

            // convert the brightness value to a 4-byte RGBA value
            Uint8 brightness = data->pixel_values[j * data->width + i];
            data->pixels[j * data->width + i] = (brightness << 24) |
                                              (brightness << 16) |
                                              (brightness << 8) |
                                              0xFF;
        }
    }
    return NULL;
}

/**
 * Iteratively calls @function choose_brightness_mandelbrot or @function choose_heatmap_mandelbrot based on
 * the choice 
 * @param renderer – the renderer in SDL which is where the mandelbrot image will be drawn
 * @param choice – decides which fractal we will use
 */
void create_fractal(SDL_Renderer* renderer, int choice) {
    // A 1-D arrray of pixels of values 0~255 where pixel ij is j * WINDOW_WIDTH + i (row major order)
    uint8_t* cpu_pixel_values = (uint8_t*)malloc(WINDOW_HEIGHT * WINDOW_WIDTH);
    uint8_t* gpu_pixel_values;

    // allocate memory for gpu_pixel_values
    if(cudaMalloc(&gpu_pixel_values, WINDOW_HEIGHT * WINDOW_WIDTH) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate pixel array on GPU\n");
        exit(2);
    }

    // we don't need to copy values to gpu because there is no value to be copied
    // gpu will compute the uint8_t values and we will pass it back to cpu_pixel_values afterwards. 

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
            gpu_pixel_values);
    } else if (choice == 2) {  // Julia set
        brightness_julia<<<numBlocks, threadsPerBlock>>>(
            WINDOW_HEIGHT, WINDOW_WIDTH, baseWidth, baseHeight, 
            centerX, centerY, zoomScale, constantA, constantB,
            maxIterations, isInfinite, gpu_pixel_values);
    } else {
        printf("Invalid choice: %d\n", choice);
        return; // exit the function if the choice is invalid
    }

    // Copy back uint8_t values
    cudaMemcpy(cpu_pixel_values, gpu_pixel_values, WINDOW_HEIGHT * WINDOW_WIDTH, cudaMemcpyDeviceToHost);

    // Create pixel buffer for SDL texture
    Uint32* pixels = (Uint32*)malloc(WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(Uint32));
    
    // Create threads
    const int num_threads = 20;  
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    
    // Calculate rows per thread
    int rows_per_thread = WINDOW_HEIGHT / num_threads + 1;
    
    // Create and launch threads
    for(int i = 0; i < num_threads; i++) {
        // set the thread data
        thread_data[i].pixel_values = cpu_pixel_values;
        thread_data[i].pixels = pixels;
        thread_data[i].width = WINDOW_WIDTH;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == num_threads - 1) ? WINDOW_HEIGHT : (i + 1) * rows_per_thread;
        
        // create the thread
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

    // Cleanup
    cudaFree(gpu_pixel_values);
    free(cpu_pixel_values);

}//end create_fractal

/**
 * @brief Scalenumber–maps a current value index of a one-dimensional field onto a larger field.
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