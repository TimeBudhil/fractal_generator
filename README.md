# Fractal Generator

This project is a final assignment for CSC213, called "Fractal Generator". The project involves generating fractals using GPU programming, CPU paralellization, and eventually networking among computers.

## Table of Contents
- [Project Overview](#project-overview)
- [Implementation Details](#implementation-details)
  - [GPU Programming With Thread](#gpu-programming)
  - [CPU With Threads](#CPU)
  - [Networking](#networking)
  - [Next Steps](#next-steps)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Instructions](#instructions)
  - [Specific Example](#Specific)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The Fractal Generator creates fractal images by leveraging GPU, CPU, and eventually Networking capabilities for computation efficiency and speed to create as smooth of an experience as possible.

## Implementation Details

### GPU Programming With Threads

The core of the fractal generation is implemented using GPU programming to take advantage of parallel processing capabilities. With each individual pixel's brightness being calculated by a individual thread. Which ensures that the data are generated efficiently and quickly.

### CPU With Threads
After the GPU is done calculating the brightness needed for each pixel, each thread of the CPU takes a couple of the pixels information and converts that brightness into a argb to be used by SDL for generating the image.

### Networking (Goal)

To make the generation even faster, our goal was to create a network so that we had access to other computers GPU's and CPU's to help generate the necessary brightness calculations and argb data before sending it back to the main computer to use the SDL library to generate the image. This is important to make our generator even faster because as you increase the number of iterations and zoom in the computer starts to slow down into a state similar to when we were only using the CPU, this is why creating a network of CPU's and GPU's would be important in preventing it from becoming slow.

On the surface, this is not that difficult because establishing a network across computers is feasible due to our prior experience on the topic in the class. However, trying to get threads to work in parallel across different computers would be a massive challenge due to the fact that our experience with networking is a little limited. We know it is possible get GPU's and CPU's to work in parallel, which would be required for threads, across different computers due to the fact that we see it in everyday life with things like crypto. However, the three of us have limited experience and we simply did not have enough time to experiment and with paralellization across different computers to pull this off, but if we had the extra time  or even just a little more experience, it could be be implemented easily.

### Next Steps

The next steps and the full stance on networking will be inside the report file.

## Requirements

- C Compiler (e.g., `gcc`)
- Make
- GPU with CUDA support
- SDL Library
  
## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/TimeBudhil/fractal_generator.git
    cd fractal_generator
    ```

2. Compile the project:
    ```sh
    make run
    ```

## Usage
### Instructions

Arrow Keys - Move the screen

I key - Increases iterations

O key - Decreases iterations

Left-click/Scroll up - Zoom in

Scross down - Zoom out

### Specific Example
After starting the program, zoom in to whatever looks cool and if you want to increase the detail or make it darker you press I but if you want it to be brighter and lose focus you press O. If you start seeing pixels, you have reached the limit of double precision and then scroll down to zoom out.
