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
  - [Example](#example)
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

To make the generation even faster, our goal was to create a network so that we had access to other computers GPU's and CPU's to help generate the necessary brightness calculations and argb data before sending it back to the main computer to use the SDL library to generate the image.

### Next Steps

networking

## Requirements

- C Compiler (e.g., `gcc`)
- Make
- GPU with CUDA support (optional but recommended for better performance)
- SDL Library
- Networking capabilities for distributed computing

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/TimeBudhil/fractal_generator.git
    cd fractal_generator
    ```

2. Compile the project:
    ```sh
    make
    ```

## Usage

To generate a fractal image, run the following command:
```sh
./fractal_generator [options]
