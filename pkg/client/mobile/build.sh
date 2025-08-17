#!/bin/bash

# CANIDAE Mobile SDK Build Script
# Builds mobile bindings for iOS and Android

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../../.."
BUILD_DIR="$SCRIPT_DIR/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🐺 CANIDAE Mobile SDK Builder${NC}"
echo "================================"

# Check for gomobile
if ! command -v gomobile &> /dev/null; then
    echo -e "${YELLOW}Installing gomobile...${NC}"
    go install golang.org/x/mobile/cmd/gomobile@latest
    go install golang.org/x/mobile/cmd/gobind@latest
    export PATH=$PATH:~/go/bin
fi

# Initialize gomobile if needed
if [ ! -d "$HOME/.cache/gomobile" ]; then
    echo -e "${YELLOW}Initializing gomobile...${NC}"
    gomobile init
fi

# Clean build directory
echo -e "${YELLOW}Cleaning build directory...${NC}"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/ios"
mkdir -p "$BUILD_DIR/android"

# Change to project root for proper module resolution
cd "$PROJECT_ROOT"

# Build for iOS
if [[ "$1" == "ios" ]] || [[ "$1" == "all" ]] || [[ -z "$1" ]]; then
    echo -e "${GREEN}Building iOS framework...${NC}"
    
    # Build for iOS (both device and simulator)
    gomobile bind \
        -target=ios \
        -o "$BUILD_DIR/ios/Canidae.xcframework" \
        github.com/macawi-ai/canidae/pkg/client/mobile
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ iOS framework built successfully${NC}"
        echo "   Output: $BUILD_DIR/ios/Canidae.xcframework"
    else
        echo -e "${RED}❌ iOS build failed${NC}"
        exit 1
    fi
fi

# Build for Android
if [[ "$1" == "android" ]] || [[ "$1" == "all" ]] || [[ -z "$1" ]]; then
    echo -e "${GREEN}Building Android AAR...${NC}"
    
    # Build for Android
    gomobile bind \
        -target=android \
        -androidapi 21 \
        -o "$BUILD_DIR/android/canidae.aar" \
        github.com/macawi-ai/canidae/pkg/client/mobile
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Android AAR built successfully${NC}"
        echo "   Output: $BUILD_DIR/android/canidae.aar"
        echo "   Sources: $BUILD_DIR/android/canidae-sources.jar"
    else
        echo -e "${RED}❌ Android build failed${NC}"
        exit 1
    fi
fi

# Generate documentation
if [[ "$1" == "docs" ]] || [[ "$1" == "all" ]]; then
    echo -e "${GREEN}Generating documentation...${NC}"
    
    cd "$SCRIPT_DIR"
    go doc -all . > "$BUILD_DIR/API.txt"
    
    echo -e "${GREEN}✅ Documentation generated${NC}"
    echo "   Output: $BUILD_DIR/API.txt"
fi

# Show build summary
echo ""
echo -e "${GREEN}🎉 Build Complete!${NC}"
echo "================================"
echo "Build artifacts:"
ls -la "$BUILD_DIR"/*/*

# Show usage instructions
echo ""
echo -e "${YELLOW}Usage Instructions:${NC}"
echo ""
echo "iOS:"
echo "  1. Drag $BUILD_DIR/ios/Canidae.xcframework into your Xcode project"
echo "  2. Import with: import Canidae"
echo ""
echo "Android:"
echo "  1. Copy $BUILD_DIR/android/canidae.aar to your app/libs folder"
echo "  2. Add to build.gradle: implementation files('libs/canidae.aar')"
echo ""
echo "See README.md for integration examples."