#!/bin/bash

kaggle_command="kaggle datasets download -d divyanshu99/spoken-digit-dataset;"
data_dir=data

eval $kaggle_command;
mkdir -p $data_dir;
mv spoken-digit-dataset.zip $data_dir;
unzip $data_dir/spoken-digit-dataset.zip -d $data_dir;
rm $data_dir/spoken-digit-dataset.zip
mv $data_dir/free-spoken-digit-dataset-master/* $data_dir/;
rm -rf $data_dir/free-spoken-digit-dataset-master/;
ls -d $data_dir/* | grep -v "recordings" | xargs rm -rf;
rm -rf $data_dir/recordings/*nicolas*;
