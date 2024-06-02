#!/bin/bash

# This script converts all .world files in DIR in the specified directory to .usd files.

DIR=$1

for file in "$DIR"/*.world; do
  if [[ -e "$file" ]]; then
    base="${file%.world}"
    output="${base}.usd"
    ./sdf2usd "$file" "$output"
    echo "Converted $file to $output"
  else
    echo "No .world files found in $DIR."
    break
  fi
done
