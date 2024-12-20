# Fractal Generator

This project is a final assignment for CSC213, titled "Fractal Generator". The project involves generating fractals using GPU programming, managing image files, and networking among computers.

## Table of Contents
- [Project Overview](#project-overview)
- [Implementation Details](#implementation-details)
  - [GPU Programming With Thread](#gpu-programming with threads)
  - [Paralellization With Threads](#Paralellization with threads)
  - [Networking](#networking)
  - [Next Steps](#next-steps)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project generates fractal images by leveraging GPU capabilities for computation efficiency. It also includes management of the generated image files and supports networking to distribute tasks among multiple computers.

## Implementation Details

### GPU Programming With Threads

The core of the fractal generation is implemented using GPU programming to take advantage of parallel processing capabilities. This ensures that fractal images are generated efficiently and quickly. The primary language used for this part is C.

### Parallelization With Threads

### Networking (Goal)

The project supports networking to allow the distribution of fractal generation tasks across multiple computers. This helps in speeding up the overall process and managing larger fractal generation tasks effectively.

### Next Steps

networking

## Requirements

- C Compiler (e.g., `gcc`)
- Make
- GPU with CUDA support (optional but recommended for better performance)
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
