# make clean
# make
# make run

# Compiler settings
CC = nvcc
CFLAGS = -g -Wall
SDL_INC = $(shell sdl2-config --cflags)
SDL_LIB = $(shell sdl2-config --libs)

# Directories
SRC_DIR = .
BUILD_DIR = build
BIN_DIR = bin

# Files
EXECUTABLE = $(BIN_DIR)/main
SOURCES = $(SRC_DIR)/main.cu
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Make sure the directories exist
$(shell mkdir -p $(BUILD_DIR) $(BIN_DIR))

# Default target
all: $(EXECUTABLE)

# Link the executable
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(SDL_LIB) -o $@ -lm

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(CC) $(CFLAGS) $(SDL_INC) -c $< -o $@

# Clean build files
clean:
	rm -rf $(BUILD_DIR)/* $(BIN_DIR)/*

# Run the program
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Rebuild everything
rebuild: clean all

.PHONY: all clean rebuild run