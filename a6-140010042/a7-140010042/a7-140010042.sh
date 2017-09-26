#!/bin/bash



echo '///////////////////////////////'
echo '//  Asignment 7 - 140010042  //'
echo '///////////////////////////////'

echo 'Simulating for Re = 1000 for 2 sec' 
python3 core.py


echo '\n'
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

