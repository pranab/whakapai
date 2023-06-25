#!/bin/bash

#build and publish package
if [[ "$#" -ne 1 ]]; then
    echo "illegal number of parameters, expecting one" 
    exit 2
fi
if [[ ! -d "$1" ]]
then
    echo "directory $1 does not exist"
    exit 2
fi
cd "$1"
echo "*** current directory ***"
pwd
echo "**** removing dist and build directories *****"
rm -rf ./dist
rm -rf ./build
echo "**** doing build ****"
python3 -m build
echo "**** publishing package ****"
python3 -m twine upload --repository testpypi dist/*
