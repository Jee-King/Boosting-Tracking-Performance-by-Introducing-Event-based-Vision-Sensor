#!/bin/bash
pth="/home/iccd/Documents/TransT-fusion/pytracking/tracking_results/transt/"
concrete="transt50"
for i in $(seq 240 249)
do
	name=`echo $i|awk '{printf("%04d\n", $0)}'`
	echo ${name}
	python run_tracker.py  transt transt50 --dataset eotb --sequence val --epochname TransT_ep${name}.pth.tar
	mv "$pth$concrete" "$pth${concrete}-e$i"
done
