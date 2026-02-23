#!/bin/bash
set -e

HUGO_VERSION="0.141.0"
HUGO_TAR="hugo_extended_${HUGO_VERSION}_linux-amd64.tar.gz"

echo "==> Downloading Hugo ${HUGO_VERSION} extended..."
curl -fsSL "https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/${HUGO_TAR}" \
  | tar -xz hugo

echo "==> Hugo version: $(./hugo version)"
./hugo --gc --minify
