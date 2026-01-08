#!/bin/bash

# BEVFormer Build Script for AX650
# This script automatically downloads all dependencies and builds the project for AX650

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Chip type is fixed to ax650 for this script
CHIP_TYPE=ax650

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_ax650"
THIRDPARTY_DIR="${SCRIPT_DIR}/3rdparty"
TOOLCHAIN_DIR="${BUILD_DIR}/toolchain"
BSP_DIR="${BUILD_DIR}/ax650n_bsp_sdk"
BSP_REPO="https://github.com/AXERA-TECH/ax650n_bsp_sdk.git"

# URLs
OPENCV_URL="https://github.com/AXERA-TECH/ax-samples/releases/download/v0.1/opencv-aarch64-linux-gnu-gcc-7.5.0.zip"
TOOLCHAIN_URL1="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
TOOLCHAIN_URL2="https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}BEVFormer ${CHIP_TYPE^^} Build${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if running on x86_64 (for cross-compilation)
ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
    echo -e "${YELLOW}Warning: Not running on x86_64, assuming native compilation${NC}"
    CROSS_COMPILE=false
else
    CROSS_COMPILE=true
fi

# Function to download file with retry and progress
download_file() {
    local url=$1
    local output=$2
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        echo "Downloading $(basename $output) (attempt $((retry + 1))/$max_retries)..."
        if wget --progress=bar:force:noscroll --show-progress -O "$output" "$url" 2>&1; then
            if [ -f "$output" ] && [ -s "$output" ]; then
                echo -e "${GREEN}✓ Downloaded: $(basename $output) ($(du -h "$output" | cut -f1))${NC}"
                return 0
            fi
        fi
        retry=$((retry + 1))
        if [ $retry -lt $max_retries ]; then
            echo -e "${YELLOW}Retry $retry/$max_retries...${NC}"
            sleep 2
            rm -f "$output" 2>/dev/null || true
        fi
    done
    
    echo -e "${RED}✗ Failed to download: $url${NC}"
    return 1
}

# Step 1: Check and install system dependencies
echo -e "\n${GREEN}[1/5] Checking system dependencies...${NC}"
MISSING_DEPS=()

command -v cmake >/dev/null 2>&1 || MISSING_DEPS+=("cmake")
command -v wget >/dev/null 2>&1 || MISSING_DEPS+=("wget")
command -v unzip >/dev/null 2>&1 || MISSING_DEPS+=("unzip")
command -v tar >/dev/null 2>&1 || MISSING_DEPS+=("tar")
command -v git >/dev/null 2>&1 || MISSING_DEPS+=("git")
command -v make >/dev/null 2>&1 || MISSING_DEPS+=("build-essential")

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Missing dependencies: ${MISSING_DEPS[*]}${NC}"
    echo -e "${YELLOW}Please install them with: sudo apt-get install ${MISSING_DEPS[*]}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ All system dependencies found${NC}"

# Step 2: Download and setup OpenCV
echo -e "\n${GREEN}[2/5] Setting up OpenCV library...${NC}"
mkdir -p "${THIRDPARTY_DIR}"
OPENCV_ZIP="${THIRDPARTY_DIR}/opencv-aarch64-linux-gnu-gcc-7.5.0.zip"
OPENCV_EXTRACT_DIR="${THIRDPARTY_DIR}/opencv-aarch64-linux"  # Actual extracted directory name

if [ ! -d "${OPENCV_EXTRACT_DIR}" ]; then
    if [ ! -f "${OPENCV_ZIP}" ]; then
        download_file "${OPENCV_URL}" "${OPENCV_ZIP}"
    fi
    
    echo "Extracting OpenCV..."
    unzip -q -o "${OPENCV_ZIP}" -d "${THIRDPARTY_DIR}"  # -o to overwrite without prompt
    echo -e "${GREEN}✓ OpenCV setup complete${NC}"
else
    echo -e "${GREEN}✓ OpenCV already exists${NC}"
fi

# Set OpenCV_DIR to the cmake config directory
if [ -d "${OPENCV_EXTRACT_DIR}/lib/cmake/opencv4" ]; then
    export OpenCV_DIR="${OPENCV_EXTRACT_DIR}/lib/cmake/opencv4"
elif [ -d "${OPENCV_EXTRACT_DIR}/share/OpenCV" ]; then
    export OpenCV_DIR="${OPENCV_EXTRACT_DIR}/share/OpenCV"
else
    export OpenCV_DIR="${OPENCV_EXTRACT_DIR}"
fi

# Step 3: Download and setup BSP SDK
echo -e "\n${GREEN}[3/5] Setting up BSP SDK for ${CHIP_TYPE}...${NC}"
if [ ! -d "${BSP_DIR}/msp/out" ]; then
    if [ ! -d "${BSP_DIR}" ]; then
        echo "Cloning BSP SDK repository for ${CHIP_TYPE}..."
        git clone --depth 1 "${BSP_REPO}" "${BSP_DIR}" || {
            echo -e "${RED}✗ Failed to clone BSP SDK for ${CHIP_TYPE}${NC}"
            echo -e "${YELLOW}Please check your network connection or download manually${NC}"
            echo -e "${YELLOW}BSP repository: ${BSP_REPO}${NC}"
            exit 1
        }
    else
        echo "Updating BSP SDK..."
        cd "${BSP_DIR}"
        git pull || echo -e "${YELLOW}Warning: Failed to update BSP SDK${NC}"
        cd "${SCRIPT_DIR}"
    fi
    
    if [ ! -d "${BSP_DIR}/msp/out" ]; then
        echo -e "${RED}✗ BSP SDK msp/out directory not found in ${BSP_DIR}${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ BSP SDK setup complete${NC}"
else
    echo -e "${GREEN}✓ BSP SDK already exists${NC}"
fi

export BSP_MSP_DIR="${BSP_DIR}/msp/out"

# Step 4: Download and setup cross-compilation toolchain (only for x86_64)
if [ "$CROSS_COMPILE" = true ]; then
    echo -e "\n${GREEN}[4/5] Setting up cross-compilation toolchain...${NC}"
    mkdir -p "${TOOLCHAIN_DIR}"
    TOOLCHAIN_TAR="${TOOLCHAIN_DIR}/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
    TOOLCHAIN_DIR_EXTRACTED="${TOOLCHAIN_DIR}/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"
    
    if [ ! -d "${TOOLCHAIN_DIR_EXTRACTED}/bin" ]; then
        if [ ! -f "${TOOLCHAIN_TAR}" ]; then
            if ! download_file "${TOOLCHAIN_URL1}" "${TOOLCHAIN_TAR}"; then
                echo "Trying alternative URL..."
                download_file "${TOOLCHAIN_URL2}" "${TOOLCHAIN_TAR}" || {
                    echo -e "${RED}✗ Failed to download toolchain${NC}"
                    exit 1
                }
            fi
        fi
        
        echo "Extracting toolchain..."
        tar -xf "${TOOLCHAIN_TAR}" -C "${TOOLCHAIN_DIR}"
        echo -e "${GREEN}✓ Toolchain setup complete${NC}"
    else
        echo -e "${GREEN}✓ Toolchain already exists${NC}"
    fi
    
    export PATH="${TOOLCHAIN_DIR_EXTRACTED}/bin:${PATH}"
    
    # Verify toolchain
    if ! aarch64-none-linux-gnu-gcc -v >/dev/null 2>&1; then
        echo -e "${RED}✗ Toolchain verification failed${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Toolchain verified${NC}"
else
    echo -e "\n${GREEN}[4/5] Skipping cross-compilation toolchain (native build)${NC}"
fi

# Step 5: Build the project
echo -e "\n${GREEN}[5/5] Building BEVFormer...${NC}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure CMake
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DBSP_MSP_DIR="${BSP_MSP_DIR}"
    -DAXERA_TARGET_CHIP="${CHIP_TYPE}"
    -DOpenCV_DIR="${OpenCV_DIR}"
    -DCMAKE_INSTALL_PREFIX=./install
)

if [ "$CROSS_COMPILE" = true ]; then
    CMAKE_ARGS+=(
        -DCMAKE_TOOLCHAIN_FILE="${SCRIPT_DIR}/toolchains/aarch64-none-linux-gnu.toolchain.cmake"
    )
    echo "Configuring CMake for cross-compilation (${CHIP_TYPE})..."
else
    echo "Configuring CMake for native build (${CHIP_TYPE})..."
fi

cmake "${CMAKE_ARGS[@]}" "${SCRIPT_DIR}" || {
    echo -e "${RED}✗ CMake configuration failed${NC}"
    exit 1
}

# Build
echo "Building..."
make -j$(nproc) || {
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
}

# Install
echo "Installing..."
make install || {
    echo -e "${YELLOW}Warning: Install failed, but build may have succeeded${NC}"
}

# Check if binary exists
BINARY_PATH=""
if [ -f "./install/bin/bevformer_inference" ]; then
    BINARY_PATH="./install/bin/bevformer_inference"
    cp "${BINARY_PATH}" ./bevformer_inference 2>/dev/null || true
elif [ -f "./bevformer_inference" ]; then
    BINARY_PATH="./bevformer_inference"
fi

if [ -n "${BINARY_PATH}" ] && [ -f "${BINARY_PATH}" ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Build completed successfully for ${CHIP_TYPE}!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "Chip type: ${CHIP_TYPE}"
    echo -e "Binary location: ${BUILD_DIR}/${BINARY_PATH}"
    echo -e "File info:"
    file "${BUILD_DIR}/${BINARY_PATH}"
    ls -lh "${BUILD_DIR}/${BINARY_PATH}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}✗ Build completed but binary not found${NC}"
    exit 1
fi
