CUDA Rasterizer
===============

[CLICK ME FOR INSTRUCTION OF THIS PROJECT](./INSTRUCTION.md)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

## Scanline Rasterizer with CUDA on the GPU
### Connie Chang
  * [LinkedIn](https://www.linkedin.com/in/conniechang44), [Demo Reel](https://www.vimeo.com/ConChang/DemoReel)
* Tested on: Windows 10, Intel Xeon CPU E5-1630 v4 @ 3.70 GHz, GTX 1070 8GB (SIG Lab)

![](renders/10-14-3-50-cow-colorInterp.PNG)  
A render of a cow with randomly assigned colors

## Introduction
Rasterization is a rendering technique that is faster than path tracing, but is not physically-based. A scanline rasterizer loops through every piece of geometry in the scene, and checks if it overlaps with a pixel. The geometry that is closest to the camera provides the color for that pixel. The technique is used widely in video games due to its speed.  

For this project, I implemented most of the graphics pipeline for a rasterizer using CUDA. The steps I coded were the vertex shader, rasterization, depth testing, and fragment shader. Depending on the step, a thread could represent a vertex, a primitive, or a pixel. The meat of the functionality is contained within the rasterizeKernel, which handles both rasterization and depth testing. To prevent race conditions during depth testing, a mutex is created for each pixel. The mutex locks the pixel's depth value if a thread is writing to it, preventing overlapping reads and writes that could lead to incorrect renders.

## Features
- Supersampling Anti-aliasing
- Color interpolation
- Textures with bilinear interpolation

## Anti-aliasing

## Color interpolation

## Textures and bilinear interpolation

## Performance Analysis
All analysis was gathered using the duck scene. 

### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
