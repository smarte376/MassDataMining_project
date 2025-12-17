#! /bin/bash

for dir in $(find results/fin/lsun*/classification_results -type d)
do
    echo $dir
    owd=$(pwd)
    cd $dir
    rm metrics.txt roc_curve.png
    sed -i 's/celeba/real/g' results.csv
    cd ..
    python ../../../metrics/get_metrics.py --results_dir . > classification_results/metrics.txt
    mv roc_curve.png ./classification_results
    cd $owd
done