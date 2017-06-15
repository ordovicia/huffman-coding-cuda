#!/bin/bash

nvcc many_elements.cu

: > tmp.log
for i in $(seq 32); do
    echo $i
    sudo ./a.out | tee -a tmp.log
done

awk 'BEGIN { ave = 0 } { ave += ($1 - ave) / NR } END { print ave }' tmp.log >> bench.log
