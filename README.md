CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Lan Lou
	*  [github](https://github.com/LanLou123).
* Tested on: Windows 10, i7-6700HQ @ 2.60GHz 16GB, GTX 1070 8GB (Personal Laptop)

### Sample Rasterization

#### ```resolution```: 900X900 ```GLTF model```: cesiummilktruck ```shader``` : blinn_phong perspective corrected bilinear textureed
![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/truck.gif)

### Introduction

  Rasterization is an efficient rendering technique commonly used in computer graphics and especially in games,simmilar to path and ray tracing, it basically does one thing: transforming the 3d object into 2d screen. 
  Different from raytracing or pathtracing, however, in rasterization, we will not track rays' further interaction with geometry anymore, instead, we will only cast the rays from each screen pixels into the scene, and get the color, depth, specular, etc results, and use these to simulate the scene, so as a consequence, rasterization is much more efficient, but is harder to get to an realistic result.

### Features:

- Basic features:
  - Vertex shading
  - Primitive assembly 
  - Rasterization
  - Fragment shading
  - A depth buffer for storing and depth testing fragments
  - Fragment-to-depth-buffer writing (with atomics for race avoidance)
  -  simple lighting scheme including togglable Lambert and Blinn-Phong
- Extra:
  - UV texture mapping with bilinear texture filtering and perspective correct texture coordinates
  - Support for rasterizing additional primitives with toggle, including line, points
  - correct color interpolation on a primitive
  - * tried SSAO, but result is not accurate
  
## Debug view:

albedo buffer|depth buffer|
------------|--------
![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/diffuse.gif) | ![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/depth.gif)  

normal buffer|specular buffer|
------------|--------
![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/normal.gif) | ![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/spec.gif)

#### combined:
![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/res.gif)

## Support for other primitives:

point|line|triangle
-----|----|-----
![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/p.gif) | ![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/line.gif) | ![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/lamm.gif)




## perspective correct texture coordinates

in the following to comparisions, the right image both shows what we will get when we use simple linear interpolation to aquire stuffs like normal, albedo, and depth, the result is apparently wrong and looks wierd, the reason for this is that when we are doing interpolation, we are only using the barycentric values in triangle vertices and the triangle value we want to interpolate, we haven't taken depth(z) information into consideration, which is really important for correctly transforming 3d data into 2d screen (depth information can't be lost), so what we should do instead is to use both the baryalue and z value to compute our result.

corrected duck|not-corrected duck
-----|----
![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/yes.gif) | ![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/non.gif)

corrected checkboard|not-corrected checkboard
-----|----
![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/perspcorrect.JPG) | ![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/notcorrected.JPG)

## correct color interpolation between points on a primitive

![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/colorinterp.gif)

# Performance analysis

## break down of pipeline time consumption:

![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/imgg.JPG)

as can be seen from the above graph, for each model, vertex transfrom almost cost the same amount of time,as it is bascically a parallel data copying process, for the same reason, primitive assembly time is the same in spite of different models, apparently, most time is used to do the rasterization operation, because we will have a lot of iterative checks for each thread, moreover, I put the bilinear filtering inside the triangle rasterization kernal, so it might bring the time consumption even higher, finally, rendering is also the same for all the models, this is simply because the shader are just too simple.....

## with and without perspective correction:

![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/pcomp.JPG)

the above image is tested using duck with and without perspective correction, it shows that, with perspective correction, we have some decrease in rasterization efficiency, this might because we have to do extra computation with z values in order to interpolate stuff.

### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
