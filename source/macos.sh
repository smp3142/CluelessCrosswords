#!/bin/sh
rm -rf build dist CluelessCrosswords.spec __pycache__
rm ../binaries/macos.zip
py2applet --make-setup CluelessCrosswords.py
python3 setup.py py2app --packages=reportlab
cp useful.pkl.gz words.pkl.gz ./dist/CluelessCrosswords.app/Contents/Resources/
rm -rf build CluelessCrosswords.spec __pycache__
rm -f setup.py
cd dist
zip -r ../../binaries/macos.zip  CluelessCrosswords.app
cd ..
