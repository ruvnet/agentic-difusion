#!/bin/bash
# AdaptDiffuser CLI Test Runner
# This script runs the AdaptDiffuser CLI tests with various parameters

# Set colors for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}    AdaptDiffuser CLI Test Runner    ${NC}"
echo -e "${BLUE}==================================================${NC}"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed or not in PATH${NC}"
    exit 1
fi

# Check if test script exists
if [ ! -f "scripts/test_adaptdiffuser_cli.py" ]; then
    echo -e "${RED}Error: Test script not found at scripts/test_adaptdiffuser_cli.py${NC}"
    exit 1
fi

# Check if config files exist
if [ ! -f "config/adaptdiffuser_cpu.yaml" ]; then
    echo -e "${RED}Error: CPU config file not found at config/adaptdiffuser_cpu.yaml${NC}"
    exit 1
fi

if [ ! -f "config/adaptdiffuser_gpu.yaml" ]; then
    echo -e "${YELLOW}Warning: GPU config file not found at config/adaptdiffuser_gpu.yaml${NC}"
fi

# Function to run tests
run_test() {
    device=$1
    echo -e "\n${GREEN}Running AdaptDiffuser tests on $device${NC}\n"
    python scripts/test_adaptdiffuser_cli.py --device "$device"
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✓ AdaptDiffuser tests on $device completed successfully${NC}"
    else
        echo -e "\n${RED}✗ AdaptDiffuser tests on $device failed${NC}"
    fi
}

# Parse command line arguments
device="cpu"
if [ $# -gt 0 ]; then
    case "$1" in
        --cpu)
            device="cpu"
            ;;
        --gpu)
            device="gpu"
            ;;
        --both)
            device="both"
            ;;
        *)
            echo -e "${YELLOW}Unknown parameter: $1. Using default (cpu)${NC}"
            ;;
    esac
fi

# Run the tests
run_test "$device"

echo -e "\n${BLUE}==================================================${NC}"
echo -e "${BLUE}    AdaptDiffuser CLI Test Complete    ${NC}"
echo -e "${BLUE}==================================================${NC}"