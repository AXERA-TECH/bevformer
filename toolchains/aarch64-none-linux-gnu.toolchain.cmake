# Cross-compilation toolchain for aarch64-none-linux-gnu
# Used for building BEVFormer on x86_64 for AX650 (aarch64)

# Set cross-compiled system type
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR aarch64)

# Set cross-compiler
SET(CMAKE_C_COMPILER   "aarch64-none-linux-gnu-gcc")
SET(CMAKE_CXX_COMPILER "aarch64-none-linux-gnu-g++")

# Set searching rules for cross-compiler
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Set pkg-config to use cross-compilation environment
SET(PKG_CONFIG_EXECUTABLE "aarch64-none-linux-gnu-pkg-config")

