#! /bin/bash

re="\/(.*)_[0-9]+_[0-9]+"
for dir in $(find results -maxdepth 1 -mindepth 1 -type d)
do
    if [[ $dir =~ $re ]]; then dataset=${BASH_REMATCH[1]}; fi
    echo "${dir}/classification_results/results.csv"
    ./parse_results.sh $dataset "${dir}/classification_results" results.csv
done