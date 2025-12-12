#! /bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <dataset> <dir> <output-filename>"
    exit 1
fi

old_wd=$(pwd)
cd $2
grep -i "^classifying" *results.txt | 
awk '{print $2,$4,$10}' |
sed -n -e 's/\/\([0-9]*\)//' -e 's/_\S*//g' -e "s/$1/real/I" -e 's/ /,/gp' |
tr '[:upper:]' '[:lower:]' > $3
cd $old_wd