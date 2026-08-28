#!/bin/bash
# Build the pbp-pmedian C kernel shared library.
# Usage: bash build_pmedian.sh
# Mirrors the tmc-pbp build script conventions.

set -e
cd "$(dirname "$0")"

CC="${CC:-cc}"
CFLAGS="-O2 -Wall -Wextra -std=c99 -fPIC"

case "$(uname -s)" in
    Darwin)
        SOEXT="dylib"
        SOFLAGS="-dynamiclib"
        ;;
    *)
        SOEXT="so"
        SOFLAGS="-shared"
        ;;
esac

$CC $CFLAGS $SOFLAGS -o "pbp_pmedian/libpmedian_core.$SOEXT" src/pmedian_core.c
echo "built pbp_pmedian/libpmedian_core.$SOEXT"
