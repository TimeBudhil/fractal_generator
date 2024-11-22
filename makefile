# make clean
# make
# make run

# Compiler settings
CC = clang
CFLAGS = -Wall -Wextra -g
SDL_INC = -I/opt/homebrew/include/SDL2
SDL_LIB = -L/opt/homebrew/lib -lSDL2

# Directories
SRC_DIR = .
BUILD_DIR = build
BIN_DIR = bin

# Files
EXECUTABLE = $(BIN_DIR)/game
SOURCES = $(SRC_DIR)/main.c
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Make sure the directories exist
$(shell mkdir -p $(BUILD_DIR) $(BIN_DIR))

# Default target
all: $(EXECUTABLE)

# Link the executable
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(SDL_LIB) -o $@

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
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