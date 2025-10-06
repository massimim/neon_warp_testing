#!/usr/bin/env bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

echo "Git Repository Status"
echo "====================="
echo ""

# List of repositories to check
REPOS=("neon" "XLB" "warp")

# Loop through each repository
for repo in "${REPOS[@]}"; do
    if [ -d "$SCRIPT_DIR/$repo/.git" ]; then
        echo "$repo:"
        echo "  Branch: $(cd "$SCRIPT_DIR/$repo" && git rev-parse --abbrev-ref HEAD)"
        echo "  Hash:   $(cd "$SCRIPT_DIR/$repo" && git rev-parse HEAD)"
        echo "  Remote: $(cd "$SCRIPT_DIR/$repo" && git remote get-url origin 2>/dev/null || echo 'No remote configured')"
        echo "  Commit: $(cd "$SCRIPT_DIR/$repo" && git log -1 --pretty=%B | head -n 1)"
        echo "  Date:   $(cd "$SCRIPT_DIR/$repo" && git log -1 --pretty=%ci)"
        if [ -z "$(cd "$SCRIPT_DIR/$repo" && git status --porcelain)" ]; then
            echo "  Status: clean"
        else
            echo "  Status: dirty"
        fi
        echo ""
    else
        echo "$repo:  Not a git repository"
        echo ""
    fi
done

