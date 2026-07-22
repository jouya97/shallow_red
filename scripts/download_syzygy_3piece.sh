#!/usr/bin/env bash
set -euo pipefail

destination="${1:-artifacts/tablebases/syzygy-3}"
mkdir -p -- "${destination}"

for material in KQvK KRvK KBvK KNvK KPvK; do
  curl -fL --retry 2 \
    "https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/${material}.rtbw" \
    -o "${destination}/${material}.rtbw"
  curl -fL --retry 2 \
    "https://tablebase.lichess.ovh/tables/standard/3-4-5-dtz/${material}.rtbz" \
    -o "${destination}/${material}.rtbz"
done

shasum -a 256 "${destination}"/*
