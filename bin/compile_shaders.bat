@echo off

set script_dir=%~dp0

glslc.exe %script_dir%..\shaders\triangle.vert -o %script_dir%..\shaders\compiled\vert.spv
glslc.exe %script_dir%..\shaders\triangle.frag -o %script_dir%..\shaders\compiled\frag.spv
pause