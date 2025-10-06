#!/usr/bin/env bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

echo "Git Repository Status"
echo "====================="
echo ""

# Check neon repository
if [ -d "$SCRIPT_DIR/neon/.git" ]; then
    echo "neon:  $(cd "$SCRIPT_DIR/neon" && git rev-parse HEAD)"
else
    echo "neon:  Not a git repository"
fi

# Check XLB repository
if [ -d "$SCRIPT_DIR/XLB/.git" ]; then
    echo "XLB:   $(cd "$SCRIPT_DIR/XLB" && git rev-parse HEAD)"
else
    echo "XLB:   Not a git repository"
fi

# Check warp repository
if [ -d "$SCRIPT_DIR/warp/.git" ]; then
    echo "warp:  $(cd "$SCRIPT_DIR/warp" && git rev-parse HEAD)"
else
    echo "warp:  Not a git repository"
fi

