#!/bin/bash 

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

glslc "$SCRIPTPATH/../shaders/triangle.vert" -o "$SCRIPTPATH/../shaders/compiled/vert.spv"
glslc "$SCRIPTPATH/../shaders/triangle.frag" -o "$SCRIPTPATH/../shaders/compiled/frag.spv"