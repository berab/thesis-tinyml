#!/bin/bash


read -p "Which dataset do you want to download? Insert 1 for dcase22, 2 for dcase20, 3 for dcase20-3class" no
case $no in
    [1]*)
            DATASET_NAME=TAU-urban-acoustic-scenes-2022-mobile-development;
            DEV_NO=6337421;
            N_ZIPS=16;
            DATA_DIR=datasets/dcase22;;
    [2]*)
            DATASET_NAME=TAU-urban-acoustic-scenes-2020-mobile-development;
            DEV_NO=3819968;
            N_ZIPS=16;
            DATA_DIR=datasets/dcase20;;
    [3]*)
            DATASET_NAME=TAU-urban-acoustic-scenes-2020-3class-development;
            DEV_NO=3670185;
            N_ZIPS=21;
            DATA_DIR=datasets/dcase20-3class;;
      *)    echo "Invalid input.";;
esac

# ----- downloading the dataset -----
counter=1
echo Downloading $DATASET_NAME dataset...
mkdir -p $DATA_DIR
while [ $counter -le $N_ZIPS ]
do
  echo "Counter value = $counter"
  wget https://zenodo.org/record/$DEV_NO/files/$DATASET_NAME.audio.$counter.zip?download=1 -O ./$DATA_DIR/audio_$counter.zip
 ((counter++))
done

wget https://zenodo.org/record/$DEV_NO/files/$DATASET_NAME.doc.zip?download=1 -O ./$DATA_DIR/doc.zip
wget https://zenodo.org/record/$DEV_NO/files/$DATASET_NAME.meta.zip?download=1 -O ./$DATA_DIR/meta.zip

mkdir ./$DATA_DIR/dataset_zip && mv ./$DATA_DIR/*.zip ./$DATA_DIR/dataset_zip
unzip ./$DATA_DIR/dataset_zip/'*.zip' -d ./$DATA_DIR/

mv ./$DATA_DIR/$DATASET_NAME/* ./$DATA_DIR/ && rm -df ./$DATA_DIR/$DATASET_NAME
mv ./$DATA_DIR/evaluation_setup ./$DATA_DIR/setup

# Deleting zip files in case storage is an issue
#rm -rf ./$DATA_DIR/*.zip
echo "The datasets are downloaded"

