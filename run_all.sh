#! /bin/bash

dataset=celeba
dg_status=UseDG

rm -rf results/fin/*

for test_set in blur clean crop jpeg noise random_combo relight
do
    results_dir=$(python run_classification_pipeline.py --test_set_dir "data/test_sets/${dataset}/balanced_small/${test_set}" --dataset $dataset \
                        --use_defense_gan --use_eigenface --use_knn \
                        | grep -i "^classification results" | awk '{print $3}')
    results_dir=$(dirname $results_dir)
    echo $results_dir
    ./parse_results.sh $dataset "${results_dir}/classification_results" results.csv
    
    new_results_dir="results/fin/${dataset}_${dg_status}_${test_set}"
    echo $new_results_dir
    mv $results_dir $new_results_dir
    
    old_wd=$(pwd)
    cd $new_results_dir
    python ../../../metrics/get_metrics.py --results_dir . > classification_results/metrics.txt
    mv roc_curve.png ./classification_results
    cd $old_wd
done

dg_status=NoDG

for test_set in blur clean crop jpeg noise random_combo relight
do
    results_dir=$(python run_classification_pipeline.py --test_set_dir "data/test_sets/${dataset}/balanced_small/${test_set}" --dataset $dataset \
                        --use_eigenface --use_knn \
                        | grep -i "^classification results" | awk '{print $3}')
    results_dir=$(dirname $results_dir)
    echo $results_dir
    ./parse_results.sh $dataset "${results_dir}/classification_results" results.csv
    
    new_results_dir="results/fin/${dataset}_${dg_status}_${test_set}"
    echo $new_results_dir
    mv $results_dir $new_results_dir
    
    old_wd=$(pwd)
    cd $new_results_dir
    python ../../../metrics/get_metrics.py --results_dir . > classification_results/metrics.txt
    mv roc_curve.png ./classification_results
    cd $old_wd
done

dataset=lsun_bedroom
dg_status=UseDG

for test_set in blur clean crop jpeg noise random_combo relight
do
    results_dir=$(python run_classification_pipeline.py --test_set_dir "data/test_sets/${dataset}/balanced_small/${test_set}" --dataset $dataset \
                        --use_defense_gan --use_knn \
                        | grep -i "^classification results" | awk '{print $3}')
    results_dir=$(dirname $results_dir)
    echo $results_dir
    ./parse_results.sh $dataset "${results_dir}/classification_results" results.csv
    
    new_results_dir="results/fin/${dataset}_${dg_status}_${test_set}"
    echo $new_results_dir
    mv $results_dir $new_results_dir
    
    old_wd=$(pwd)
    cd $new_results_dir
    python ../../../metrics/get_metrics.py --results_dir . > classification_results/metrics.txt
    mv roc_curve.png ./classification_results
    cd $old_wd
done

dg_status=NoDG

for test_set in blur clean crop jpeg noise random_combo relight
do
    results_dir=$(python run_classification_pipeline.py --test_set_dir "data/test_sets/${dataset}/balanced_small/${test_set}" --dataset $dataset \
                        --use_knn \
                        | grep -i "^classification results" | awk '{print $3}')
    results_dir=$(dirname $results_dir)
    echo $results_dir
    ./parse_results.sh $dataset "${results_dir}/classification_results" results.csv
    
    new_results_dir="results/fin/${dataset}_${dg_status}_${test_set}"
    echo $new_results_dir
    mv $results_dir $new_results_dir
    
    old_wd=$(pwd)
    cd $new_results_dir
    python ../../../metrics/get_metrics.py --results_dir . > classification_results/metrics.txt
    mv roc_curve.png ./classification_results
    cd $old_wd
done