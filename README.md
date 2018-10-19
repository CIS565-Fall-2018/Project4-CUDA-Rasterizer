CUDA Rasterizer
===============


**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**



Hannah Bollar: [LinkedIn](https://www.linkedin.com/in/hannah-bollar/), [Website](http://hannahbollar.com/)



Tested on: Windows 10 Pro, i7-6700HQ @ 2.60GHz 15.9GB, GTX 980M (Personal)

____________________________________________________________________________________

![Developer](https://img.shields.io/badge/Developer-Hannah-0f97ff.svg?style=flat) ![CUDA 8.0](https://img.shields.io/badge/CUDA-8.0-yellow.svg) ![Built](https://img.shields.io/appveyor/ci/gruntjs/grunt.svg) ![Issues](https://img.shields.io/badge/issues-none-green.svg)

[//]: # ( ![Progress](https://img.shields.io/badge/implementation-in%20progress-orange.svg)

[Visuals](#visuals) - [Optimizations](#optimizations) - [Bloopers](#bloopers) - [References](#references) 

# Rasterizer

### All Current Features

The flags of all toggleable features can be updated in `rasterize.cu`.

Graphics Features
- Vertex Shading
- Primitive Assembly
	- Triangles
	- Lines*
	- Points*
- Rasterization
- Fragment Shading
	- Lambert
	- Regular Texture Mapping*
	- Bilinear Interpolated Texturing*
- Use of Depth Buffer for Frag Comparisons
	- Mutex to prevent race conditions
- BackFace Culling*

`*` = additional features.

### Visuals

Triangles | Normals | Lines | Points
:-------------------------:|:-------------------------:|:-------------------------:|:---------------------:
![](images/duck_tex.png)| ![](images/duck_nor.png)| ![](images/duck_lines.png)| ![](images/duck_points.png)|
![](images/cow_tex.png)| ![](images/cow_nor.png)| ![](images/cow_lines.png)| ![](images/cow_points.png)|
![](images/milktruck_tex.png)| ![](images/milktruck_nor.png)| ![](images/milktruck_lines.png)| ![](images/milktruck_points.png)

### Optimizations

![](images/raster_runtimes.png)

![](images/raster_runtimes_data.png)

![](images/trivslinevspoint.png)

![](images/trivslinevspoint_data.png)

### Bloopers

Bad Back Face Culling | Improper Line Rasterizing | Overblown Bilinear Texturing
:-------------------------:|:-------------------------:|:-------------------------:
![](images/incorrectbackface.png)| ![](images/linerasterizingwrong.png)| ![](images/correcttexturing_shadedglaretoolarge.png)

- bad back face culling - my directional check was reversed mistakenly
- improper line rasterizing blooper - I had clamped to an int to late, so my indexing was out of the appropriate bounds after I had already done my bounds check.
- overblown bilinear texturing - never properly casted to int, yielded same mathematical indexing error as improper line rasterizing blooper but visual was different because of differeing visualization scheme

### References

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
* [bilinear filtering](https://en.wikipedia.org/wiki/Bilinear_filtering)
