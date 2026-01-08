#!/bin/bash

# Script to apply patches to submodules
# This script iterates through the patches/ directory and applies each .patch file
# to its corresponding submodule directory in the project root.

# Get the root directory of the project (one level up from this script)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCHES_DIR="$PROJECT_ROOT/patches"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}${BOLD}=== Submodule Patch Application Utility ===${NC}"
echo -e "${BLUE}Working Directory: ${NC}$PROJECT_ROOT"
echo -e "${BLUE}Patches Directory: ${NC}$PATCHES_DIR"

# Check if patches directory exists
if [ ! -d "$PATCHES_DIR" ]; then
    echo -e "${RED}Error: Patches directory not found at $PATCHES_DIR${NC}"
    exit 1
fi

# Find all .patch files
patches=($(ls "$PATCHES_DIR"/*.patch 2>/dev/null))

if [ ${#patches[@]} -eq 0 ]; then
    echo -e "${YELLOW}No patch files found in $PATCHES_DIR${NC}"
    exit 0
fi

applied_count=0
skipped_count=0
failed_count=0

for patch_path in "${patches[@]}"; do
    patch_file=$(basename "$patch_path")
    submodule_name="${patch_file%.patch}"
    submodule_dir="$PROJECT_ROOT/$submodule_name"

    echo -e "\n${BLUE}Targeting submodule: ${YELLOW}${BOLD}$submodule_name${NC}"

    if [ ! -d "$submodule_dir" ]; then
        echo -e "${RED}  [!] Directory not found: $submodule_dir. Skipping.${NC}"
        ((skipped_count++))
        continue
    fi

    # 1. Check if patch is already applied (using --reverse --check)
    if git -C "$submodule_dir" apply --reverse --check "$patch_path" > /dev/null 2>&1; then
        echo -e "  [${GREEN}✓${NC}] Patch is already applied."
        ((skipped_count++))
        continue
    fi

    # 2. Check if patch can be applied cleanly
    if git -C "$submodule_dir" apply --check "$patch_path" > /dev/null 2>&1; then
        echo -e "  [ ] Applying patch..."
        if git -C "$submodule_dir" apply "$patch_path"; then
            echo -e "  [${GREEN}✓${NC}] Patch applied successfully."
            ((applied_count++))
        else
            echo -e "  [${RED}✗${NC}] Failed to apply patch during execution."
            ((failed_count++))
        fi
    else
        echo -e "  [${RED}✗${NC}] Patch cannot be applied cleanly (conflicts or files missing)."
        echo -e "      Try checking the state of ${YELLOW}$submodule_name${NC} manually."
        ((failed_count++))
    fi
done

echo -e "\n${BLUE}${BOLD}=== Summary ===${NC}"
echo -e "${GREEN}Applied: $applied_count${NC}"
echo -e "${YELLOW}Skipped: $skipped_count${NC}"
echo -e "${RED}Failed:  $failed_count${NC}"
echo -e "${BLUE}${BOLD}===============${NC}"
