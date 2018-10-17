CUDA Rasterizer
===============

[CLICK ME FOR PROJECT INSTRUCTIONS](./INSTRUCTION.md)

* Salaar Kohari
  * LinkedIn ([https://www.linkedin.com/in/salaarkohari](https://www.linkedin.com/in/salaarkohari))
  * Website ([http://salaar.kohari.com](http://salaar.kohari.com))
  * University of Pennsylvania, CIS 565: GPU Programming and Architecture
* Tested on: Windows 10, Intel Xeon @ 3.7GHz 32GB, GTX 1070 8GB (SIG Lab)

### Introduction
My GPU rasterizer produces real-time, interactive renders of complex geometry. Scene data is loaded from gltf file format and rendered using the pipeline outlined below.

### Pipeline
1. Vertex shading
2. Primitive assembly with support for triangles read from buffers of index and vertex data
3. Scanline rasterization
4. Fragment shading with lighting scheme
5. Depth buffer for storing and depth testing fragments
6. Fragment-to-depth-buffer writing (with atomics for race avoidance)

### Images
BILINEAR TEXTURE

ANTIALIASING

VERTEX COLOR INTERP

### Analysis
