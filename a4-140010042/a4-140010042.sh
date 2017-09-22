#!/bin/bash
echo '///////////////////////////////'
echo '//  Asignment 4 - 140010042  //'
echo '///////////////////////////////'
python3 Q1.py
echo 'Part 1 is completed'
echo 'Part 2 has started ...It may take some time'
python3 Q2.py

python3 Q3.py
echo 'Part 3 has started ...It may take some time'
echo 'Results are generated'
echo 'Creating report'
pdflatex -interaction=errorstopmode report.tex
pdflatex -interaction=errorstopmode report.tex
echo '**********    cleaning up    **********'
rm -rf report.aux
rm -rf report.lof
rm -rf report.log
rm -rf report.toc 
rm -rf __pycache__
echo '////////////////////////////////'
echo '// The job has been completed //'
echo '// Thankyou for your patience //'
echo '////////////////////////////////'


