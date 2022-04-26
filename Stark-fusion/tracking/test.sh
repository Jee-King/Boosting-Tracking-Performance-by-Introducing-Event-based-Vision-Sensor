#!/bin/bash
pth="/home/iccd/Documents/Stark-fusion/lib/test/tracking_results/stark_s/"
concrete="baseline"
for i in $(seq 167 179)
do
	name=`echo $i|awk '{printf("%04d\n", $0)}'`
	echo ${name}
	CUDA_VISIBLE_DEVICES=0 python test.py  stark_s baseline --dataset eotb --sequence val --epochname STARKS_ep${name}.pth.tar
	mv "$pth$concrete" "$pth${concrete}-e$i"
done
