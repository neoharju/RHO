#!/usr/bin/env bash
#
# Downloading mnist dataset using PyTorch mirror
# since yann.lecun.com/exdb/mnist failed
set -euo pipefail

URL=https://ossci-datasets.s3.amazonaws.com/mnist
DEST=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data

# filename, gz_size, gz_sha256, raw_size, raw_sha256
# gz values are taken from the mirror; raw values are of the files unpacked from them
FILES=(
  "train-images-idx3-ubyte.gz  9912422  440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609  47040016  ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db"
  "train-labels-idx1-ubyte.gz    28881  3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c     60008  65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5"
  "t10k-images-idx3-ubyte.gz   1648877  8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6   7840016  0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7"
  "t10k-labels-idx1-ubyte.gz      4542  f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6     10008  ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2"
)

error() {
	printf 'mnist check failed: %s\n' "$1" >&2; 
	exit 1;
}

# make dir /data if it does not exist
mkdir -p "$DEST"
cd "$DEST"

for file in "${FILES[@]}"; do
    read -r name gz_size sha raw_size raw_sha <<<"$file"
    uncompressed_name=${name%.gz}

    # check if we have the uncompressed file already
    # and it matches the pinned raw_size and raw_sha
    if [[ -f "$uncompressed_name" ]]; then
        actual_size=$(stat -c%s "$uncompressed_name")
        [[ "$actual_size" == "$raw_size" ]] ||
            error "$uncompressed_name is $actual_size bytes, expected $raw_size"
        actual_raw_sha=$(sha256sum "$uncompressed_name" | cut -d' ' -f1)
        [[ "$raw_sha" == "$actual_raw_sha" ]] ||
            error "sha256 mismatch: $uncompressed_name"
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

    # check the unpacked file too
    actual_raw_sha=$(sha256sum "$uncompressed_name" | cut -d' ' -f1)
    [[ "$raw_sha" == "$actual_raw_sha" ]] ||
        error "sha256 mismatch: $uncompressed_name"
done

printf 'MNIST downloaded to %s\n' "$DEST"
