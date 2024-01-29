#!/bin/bash
a=./new_collected/
b=./collected/collected/
for i in `ls $a`
do
echo $i
ls -l $b/$i
rm -rf $b/$i
done