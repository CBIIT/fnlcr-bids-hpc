#!/bin/bash

# Export from Access to Unicode (UTF-8) text format
# Use the export-tron saved export from Access if there, otherwise it's easy to do manually

# Input
text_file="/Users/weismanal/links/network_drives/network-home/tron.txt"

# Convert from DOS to UNIX per https://en.wikipedia.org/wiki/Newline#Conversion_utilities
awk '{gsub(/\r/,""); print;}' $text_file > pretty_version.txt

# Replace three newlines with two newlines until there are no more sequences of three newlines left
is_different=1
while [ $is_different -eq 1 ]; do
    echo "In loop"
    awk -v RS="XXXX" '{gsub(/\n\n\n/,"\n\n"); print}' pretty_version.txt > tmp.txt
    cmp pretty_version.txt tmp.txt && is_different=0 || is_different=1
    mv -f tmp.txt pretty_version.txt
done

# Insert spaces after the colons after the dates
# Also remove leading spaces preceding the dates... the number of spaces is fixed as seen from code like "grep "/2018:\ \|/2019:\ " pretty_version.txt | less"
# It's actually tricky to fix this in Access so that's why I'm doing these things here
mv pretty_version.txt tmp.txt
awk '{gsub("/2018:","/2018: "); gsub("/2019:","/2019: "); gsub(/^                 /,""); print}' tmp.txt > pretty_version.txt