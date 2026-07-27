#!/usr/bin/env bash
#
# Asserts that every third-party GitHub Action is pinned to a full commit SHA
# and labelled with a bare version.
#
# The SHA is the security property: a tag can be repointed by anyone who can
# write to the action's repository, and every workflow using it then runs the
# new code with nothing changed here to review.
#
# The comment's *form* is checked too, not merely its presence. Dependabot
# rewrites the trailing comment along with the SHA only when it is a bare
# version; put a reason or anything else after it and the label is silently left
# behind when the SHA moves. A comment claiming v6 beside a SHA that is now v7
# is worse than no comment, and it is exactly what a presence-only check lets
# through.

set -euo pipefail

workflows_dir="$(dirname "$0")/../.github/workflows"
status=0

fail() {
    printf '%s:%s: %s\n    %s\n' "$1" "$2" "$3" "$4" >&2
    status=1
}

while IFS= read -r workflow; do
    line_number=0
    while IFS= read -r line; do
        line_number=$((line_number + 1))

        # Only `uses:` lines naming an external action.
        [[ "$line" =~ ^[[:space:]]*-?[[:space:]]*uses:[[:space:]]* ]] || continue
        reference="${line#*uses:}"
        reference="${reference#"${reference%%[![:space:]]*}"}"

        # Actions in this repository are already at the commit under test.
        [[ "$reference" == ./* ]] && continue

        if [[ ! "$reference" =~ ^[^@[:space:]]+@[0-9a-f]{40}([[:space:]]|$) ]]; then
            fail "$workflow" "$line_number" "not pinned to a full 40-character commit SHA" "$line"
            continue
        fi

        if [[ ! "$reference" =~ @[0-9a-f]{40}[[:space:]]+\#[[:space:]]*v?[0-9]+(\.[0-9]+)*$ ]]; then
            fail "$workflow" "$line_number" \
                "trailing comment must be the bare version and nothing else" "$line"
        fi
    done < "$workflow"
done < <(find "$workflows_dir" -type f \( -name '*.yml' -o -name '*.yaml' \) | sort)

if [ "$status" -eq 0 ]; then
    echo "All third-party actions are pinned to a SHA and labelled with a bare version."
fi

exit "$status"
