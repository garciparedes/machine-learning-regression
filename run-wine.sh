#! /bin/bash

CALLING_PATH=`pwd -P`
SCRIPT_PATH="`dirname \"$0\"`"
SRC_PATH=${SCRIPT_PATH}/src
MAIN_SCRIPT="./main.sh"

echo '--------------------------------------------------------------------------------'
echo 'Title: Multiple Linear Regresion and Logistc Regresion over Wine Dataset'
echo 'Author: Sergio García Prado (garciparedes.me)'
echo 'Author: Óscar Fernández Angulo'
echo '--------------------------------------------------------------------------------'
cd ${SRC_PATH}
  octave Wine.m --no-gui
cd ${CALLING_PATH}
echo '--------------------------------------------------------------------------------'
