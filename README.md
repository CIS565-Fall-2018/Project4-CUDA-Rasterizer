CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

## Scanline Rasterizer with CUDA on the GPU
### Connie Chang
  * [LinkedIn](https://www.linkedin.com/in/conniechang44), [Demo Reel](https://www.vimeo.com/ConChang/DemoReel)
* Tested on: Windows 10, Intel Xeon CPU E5-1630 v4 @ 3.70 GHz, GTX 1070 8GB (SIG Lab)

![](renders/10-15-02-10-duck-texture.PNG)  
A render of a duck with a diffuse texture

## Introduction
Rasterization is a rendering technique that is faster than path tracing, but is not physically-based. A scanline rasterizer loops through every piece of geometry in the scene, and checks if it overlaps with a pixel. The geometry that is closest to the camera provides the color for that pixel. The technique is used widely in video games due to its speed.  

For this project, I implemented most of the graphics pipeline for a rasterizer using CUDA. The steps I coded were the vertex shader, rasterization, depth testing, and fragment shader. Depending on the step, a thread could represent a vertex, a primitive, or a pixel. The meat of the functionality is contained within the rasterizeKernel, which handles both rasterization and depth testing. To prevent race conditions during depth testing, a mutex is created for each pixel. The mutex locks the pixel's depth value if a thread is writing to it, preventing overlapping reads and writes that could lead to incorrect renders.

## Features
- Supersampling Anti-aliasing
- Color interpolation
- Textures with bilinear interpolation

## Anti-aliasing
![](renders/10-13-04-23-tri-noAA.PNG)
![](renders/10-13-04-22-tri-aa.PNG)
Images without and with 2x anti-aliasing  

Supersampling anti-aliasing was implemented to improve the visual result. The implementation is typical, rendering the image at a larger resolution before shrinking it down to the desired image side. It was tested with 2x anti-aliasing with each dimension of the image doubled. The results are shown below.  

![](renders/aa-comparison.png)  
Zoomed in comparison without and with anti-aliasing  

## Color interpolation
![](renders/10-14-3-50-cow-colorInterp.PNG)  
A render of a cow with randomly assigned colors  



## Textures and bilinear interpolation
![](renders/10-14-12-58-checkerboard-bilinear.PNG)  
A checkerboard texture using bilinear interpolation

## Performance Analysis
All analysis was gathered using the duck scene.  

Most kernels run within a reasonable time. The bulk of the logic is in the rasterize kernel, so naturally it takes the longest. Unsurprisingly, anti-aliasing uses significantly more time for rasterize and render because it is creating an image that is four times larger. sendImageToPBO also takes longer for anti-aliasing because it needs to compress the larger image into a small one.  

![](renders/Average_Kernel_Runtimes.png)  
The average runtime of each kernel for different features  

The graph below shows the same information in a stacked graph. From here, we see that texture calculation is barely slower than the without textures. This shows that textures with bilinear interpolation are a good addition to a rasterizer because it provides a huge visual improvement without much overhead.  

![](renders/Average_Runtime_Optimization.png)  
The average runtime of each feature  


### Credits

* Emily Vo for helping debug texture colors
* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
