#!/bin/bash

SRC="/mnt/c/Users/siran/ML/Deep-Learning4/deep-learning-from-scratch-4"
DST="/mnt/c/Users/siran/ML/Deep-Learning4/Deep-Learning4"

find "$SRC" -type f -print0 |
while IFS= read -r -d '' file
do
    newfile="${file/$SRC/$DST}"

    mkdir -p "$(dirname "$newfile")"
    touch "$newfile"
done