#!/usr/bin/env python

import glob
import os
import pathlib

script_dir = os.path.dirname(os.path.realpath(__file__));
shader_dir = os.path.join(script_dir, "../shaders/src");
output_dir = os.path.join(script_dir, "../shaders/compiled");


def main():
    shaders = list_shaders();
    compile_shaders(shaders);


def list_shaders():
    files = glob.glob(shader_dir + "/**/*.*", recursive=True);
    relative_files = [];
    for file in files:
        if not os.path.isfile(file): continue;
        relative_files.append(file[len(shader_dir):]);
        
    return relative_files;

def compile_shaders(shaders):
    for shader in shaders:
        pathlib.Path(output_dir + os.path.dirname(shader)).mkdir(parents=True, exist_ok=True)
        os.system("glslc " + shader_dir + shader + " -o " + output_dir + shader + ".spv")

if __name__ == "__main__":
    main();