#!/bin/bash

# Produce results from  "precomputed" folder
# ATTENTION: may take several hours

languages=("english" "german" "latin" "swedish")
embeddings=("ELMo" "BERT" "XLM-R")
k_values=(-1 3 4 5)

for lang in "${languages[@]}"; do
    for embed in "${embeddings[@]}"; do
        for k in "${k_values[@]}"; do
            python3 src/fma.py -l "$lang" -e "$embed" -k "$k"
        done
    done
done

for lang in "${languages[@]}"; do
    for embed in "${embeddings[@]}"; do
        for k in "${k_values[@]}"; do
            python3 src/pha.py -l "$lang" -e "$embed" -k "$k"
        done
    done
done

for lang in "${languages[@]}"; do
    for embed in "${embeddings[@]}"; do
        python3 src/mra.py -l "$lang" -e "$embed"
    done
done