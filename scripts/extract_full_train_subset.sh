#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_URL="https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/extract_full_train_subset.sh --confirm-240gb-download [PATH_LIST] [DESTINATION]

Defaults:
  PATH_LIST    data/metadata/full_train_paths.txt
  DESTINATION data/raw

The official server exposes one 240 GB gzip archive, not individual images.
This command streams the whole archive through tar but stores only listed files.
Run it on Katana Data Mover (KDM), not a login node.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  --confirm-240gb-download)
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
shift

cd "$PROJECT_ROOT"
PATH_LIST="${1:-data/metadata/full_train_paths.txt}"
DESTINATION="${2:-data/raw}"

if [[ ! -s "$PATH_LIST" ]]; then
  echo "Path list is missing or empty: $PATH_LIST" >&2
  echo "Run this first:" >&2
  echo "  python scripts/build_full_train_subset.py" >&2
  exit 1
fi

for command_name in curl tar awk df mktemp; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done

if awk '
  NF == 0 ||
  $0 !~ /^train\// ||
  $0 ~ /^\// ||
  $0 ~ /(^|\/)\.\.(\/|$)/ {
    print "Unsafe or invalid archive path at line " NR ": " $0 > "/dev/stderr"
    exit 1
  }
' "$PATH_LIST"; then
  :
else
  exit 1
fi

mkdir -p "$DESTINATION"
PENDING_LIST="$(mktemp)"
trap 'rm -f "$PENDING_LIST"' EXIT

while IFS= read -r archive_path; do
  if [[ ! -f "$DESTINATION/$archive_path" ]]; then
    printf '%s\n' "$archive_path" >> "$PENDING_LIST"
  fi
done < "$PATH_LIST"

TOTAL_COUNT="$(wc -l < "$PATH_LIST" | tr -d ' ')"
PENDING_COUNT="$(wc -l < "$PENDING_LIST" | tr -d ' ')"

echo "Requested additional images: $TOTAL_COUNT"
echo "Images still missing:         $PENDING_COUNT"
echo "Destination filesystem:"
df -h "$DESTINATION"

if [[ "$PENDING_COUNT" == "0" ]]; then
  echo "All requested images are already present; nothing to extract."
  exit 0
fi

echo
echo "Starting the official 240 GB archive stream."
echo "Only the $PENDING_COUNT listed images will be stored under $DESTINATION/train/."
curl --fail --location --progress-bar "$ARCHIVE_URL" |
  tar -xzf - -C "$DESTINATION" -T "$PENDING_LIST"

MISSING_AFTER=0
while IFS= read -r archive_path; do
  if [[ ! -f "$DESTINATION/$archive_path" ]]; then
    MISSING_AFTER=$((MISSING_AFTER + 1))
  fi
done < "$PATH_LIST"

if [[ "$MISSING_AFTER" -ne 0 ]]; then
  echo "Extraction finished but $MISSING_AFTER requested images are missing." >&2
  exit 1
fi

echo "Extraction verified: all $TOTAL_COUNT additional images are present."
