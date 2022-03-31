#!/usr/bin/env bash

# Abort on error
set -e

DATASET_PATH=data
echo -e "\nDownloading and decompressing Replica to $1. The script can resume\npartial downloads -- if your download gets interrupted, simply run it again.\n"

for p in {a..q}
do
  # Ensure files are continued in case the script gets interrupted halfway through
  wget --continue https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.parta$p
done

# Create the destination directory if it doesn't exist yet
mkdir -p $DATASET_PATH

cat replica_v1_0.tar.gz.part?? | unpigz -p 32  | tar -xvC $DATASET_PATH
