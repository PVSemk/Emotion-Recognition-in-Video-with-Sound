#!/bin/bash

# Usage: ./extract_audio.sh path/to/mp4/files path/to/save/output/wav

searchPath="$1/*"

for i in ${searchPath};
do
  name=`echo "$i" | cut -d'.' -f1 | cut -d'/' -f2` 
  echo "$name"
  ffmpeg -i "$i" -vn -acodec pcm_s16le "$2/${name}.wav"
done
