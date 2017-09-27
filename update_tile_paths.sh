#!/bin/bash

if [ -e $(which fileroot) ]; then
    echo "fileroot not found, install wcstools"
    exit 1
fi
labfile=$1 # e.g., ./tiles_csk/192/labels_01v23.txt
oldpath=$2 # e.g., /Users/bbue/Research/ARIA/ariamh/ariaml/tiles_csk/
newpath=$3 # e.g., ./tiles_csk/

if [ $# == 3 ]; then
    outfile="$(fileroot $labfile)_new.${labfile##*.}" 
elif [ $# == 4 ]; then 
    outfile=$4 # e.g., ./tiles_csk/192/labels_01v23_new.txt    
else
    echo "usage: $0 infile.ext oldpath newpath [outfile=infile_new.ext]"
    exit 1
fi

echo "replacing '$oldpath' with '$newpath' in $labfile"
echo "saving output to $outfile"
cat $labfile | sed "s:$oldpath:$newpath:" > $outfile
echo "done"

