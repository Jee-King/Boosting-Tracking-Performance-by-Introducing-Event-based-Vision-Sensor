#!/bin/bash
pth="/home/iccd/Document/Documents/pytracking_lstmsub/pytracking/tracking_results/dimp/"
concrete="prdimp18"
for i in $(seq 6 30)
do
	name=`echo $i|awk '{printf("%04d\n", $0)}'`
	echo ${name}
	python run_tracker.py  dimp prdimp18 --dataset eotb --sequence val --epochname DiMPnet_ep${name}.pth.tar
	mv "$pth$concrete" "$pth${concrete}-e$i"
done
