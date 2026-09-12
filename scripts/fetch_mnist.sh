#!/usr/bin/env bash
#
# Downloading mnist dataset using PyTorch mirror
# since yann.lecun.com/exdb/mnist failed
set -euo pipefail

URL=https://ossci-datasets.s3.amazonaws.com/mnist
DEST=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data

# filename, gz_size, sha256, raw_size; taken from mirror
FILES=(
  "train-images-idx3-ubyte.gz  9912422  440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609  47040016"
  "train-labels-idx1-ubyte.gz    28881  3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c     60008"
  "t10k-images-idx3-ubyte.gz   1648877  8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6   7840016"
  "t10k-labels-idx1-ubyte.gz      4542  f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6     10008"
)

error() {
	printf 'mnist check failed: %s\n' "$1" >&2; 
	exit 1;
}

# make dir /data if it does not exist
mkdir -p "$DEST"
cd "$DEST"

for file in "${FILES[@]}"; do
    read -r name gz_size sha raw_size <<<"$file"
    uncompressed_name=${name%.gz}

    # check if we have the uncompressed file already
    # and it matches the pinned raw_size
    if [[ -f "$uncompressed_name" ]]; then
        actual_size=$(stat -c%s "$uncompressed_name")
        [[ "$actual_size" == "$raw_size" ]] ||
            error "$uncompressed_name is $actual_size bytes, expected $raw_size"
        printf ' %-28s OK\n' "$uncompressed_name"
        continue
    fi

    # check if we have the compressed file
    # if not download with curl
	printf 'Downloading %s\n' "$name"
    if [[ ! -f "$name" ]]; then
        curl -fSL --retry 3 -o "$name" "$URL/$name" ||
            error "download failed: $name"
    fi

    # check the downloaded size against the pinned size
    actual_size=$(stat -c%s "$name")
    [[ "$actual_size" == "$gz_size" ]] ||
        error "$name is $actual_size bytes, expected $gz_size"

    # check SHA-256
    actual_sha=$(sha256sum "$name" | cut -d' ' -f1)
    [[ "$sha" == "$actual_sha" ]] ||
        error "sha256 mismatch: $name"

    gunzip -k "$name"
done

printf 'MNIST downloaded to %s\n' "$DEST"
