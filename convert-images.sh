#!/bin/bash
for deck in data/dobble_*55 ; do
    for img in $deck/*/*.tif ; do
        target_img=`echo $img | sed -E 's/.*(deck[0-9]+).*(card[0-9]+_[0-9]+).*/\1_\2.jpg/'`
        target_img="data/dobble/images/$target_img"
        echo "$img -> $target_img"
        convert $img $target_img
    done 
done