#!/bin/bash
#
# Everything CI runs, in one command, locally.

set -uo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

declare -a names=()
declare -a statuses=()

run() {
    local name="$1"
    shift
    echo ""
    echo "--- $name"
    "$@"
    names+=("$name")
    statuses+=("$?")
}

run "action pins" ./scripts/check-action-pins.sh
run "fmt" cargo fmt --all -- --check
run "clippy" cargo clippy --all-targets --all-features -- -Dclippy::all
run "test" cargo test --all-features
run "test (release)" cargo test --release --all-features
run "docs" cargo doc --no-deps
# The wider accuracy sweep. In CI this is scheduled rather than per-commit; it
# is worth the wait before pushing anything that touches model selection.
run "accuracy sweep" cargo test --release --test accuracy -- --ignored

echo ""
failed=0
for i in "${!names[@]}"; do
    if [ "${statuses[$i]}" -ne 0 ]; then
        echo -e " - ${names[$i]} ${RED}FAILED${NC}"
        failed=1
    fi
done

if [ "$failed" -eq 0 ]; then
    echo -e "All checks passed ${GREEN}ok${NC}."
else
    exit 1
fi
