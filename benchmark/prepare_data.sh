#!/bin/bash

cd dataset
rm -rf ILSVRC2012
wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_little_train.tar
tar xf whole_chain_little_train.tar
ln -s whole_chain_little_train ILSVRC2012
cd ILSVRC2012 
mv train.txt train_list.txt
mv val.txt val_list.txt
cd ../../
