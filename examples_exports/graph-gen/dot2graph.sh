#!/bin/sh
if [ ! -f "$1" ]; then
   echo "usage: $0 <file.dot>";
   exit 1
fi

base=`basename "$1" .dot`
out="$base.graphml"

./dottoxml/dottoxml.py --at=normal "$1" "$out"
