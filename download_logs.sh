#!/bin/bash
# Download all log files referenced in research.md from the modded-nanogpt repo

REPO="pranay5255/modded-nanogpt"
OUTPUT_DIR="logs"
BRANCH="master"

mkdir -p "$OUTPUT_DIR"

# Extract all log file paths from research.md
# Matches patterns like [log](records/...) and also [log](records/.../dir) (directory logs)
log_paths=$(grep -oP '\[log\]\(\K[^)]+' research.md | sort -u)

echo "Found $(echo "$log_paths" | wc -l) log references"
echo "================================================"

downloaded=0
failed=0
skipped=0

for path in $log_paths; do
    # Skip if it's not a records/ path
    if [[ ! "$path" == records/* ]]; then
        echo "SKIP (not a records path): $path"
        ((skipped++))
        continue
    fi

    # Create the local directory structure
    local_dir="$OUTPUT_DIR/$(dirname "$path")"
    mkdir -p "$local_dir"

    local_file="$OUTPUT_DIR/$path"

    # Check if already downloaded
    if [[ -f "$local_file" ]]; then
        echo "EXISTS: $path"
        ((skipped++))
        continue
    fi

    echo -n "Downloading: $path ... "

    # Try to download as a file using gh api
    # The GitHub API endpoint for raw file content
    encoded_path=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$path', safe=''))")

    response=$(gh api \
        "repos/$REPO/contents/$path" \
        --jq '.download_url' \
        2>/dev/null)

    if [[ -n "$response" && "$response" != "null" ]]; then
        # It's a file, download it
        if curl -sL "$response" -o "$local_file" 2>/dev/null; then
            echo "OK"
            ((downloaded++))
        else
            echo "FAIL (curl error)"
            ((failed++))
        fi
    else
        # Might be a directory - try listing contents
        dir_contents=$(gh api \
            "repos/$REPO/contents/$path" \
            --jq '.[].download_url' \
            2>/dev/null)

        if [[ -n "$dir_contents" ]]; then
            echo "(directory)"
            mkdir -p "$OUTPUT_DIR/$path"
            while IFS= read -r file_url; do
                if [[ -n "$file_url" && "$file_url" != "null" ]]; then
                    filename=$(basename "$file_url")
                    local_subfile="$OUTPUT_DIR/$path/$filename"
                    echo -n "  Downloading: $filename ... "
                    if curl -sL "$file_url" -o "$local_subfile" 2>/dev/null; then
                        echo "OK"
                        ((downloaded++))
                    else
                        echo "FAIL"
                        ((failed++))
                    fi
                fi
            done <<< "$dir_contents"
        else
            echo "FAIL (not found)"
            ((failed++))
        fi
    fi

    # Small delay to avoid rate limiting
    sleep 0.3
done

echo ""
echo "================================================"
echo "Done! Downloaded: $downloaded | Failed: $failed | Skipped: $skipped"
echo "Logs saved to: $OUTPUT_DIR/"
