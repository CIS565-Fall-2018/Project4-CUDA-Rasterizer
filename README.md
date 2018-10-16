CUDA Rasterizer
===============

[CLICK ME FOR PROJECT INSTRUCTIONS](./INSTRUCTION.md)

* Salaar Kohari
  * LinkedIn ([https://www.linkedin.com/in/salaarkohari](https://www.linkedin.com/in/salaarkohari))
  * Website ([http://salaar.kohari.com](http://salaar.kohari.com))
  * University of Pennsylvania, CIS 565: GPU Programming and Architecture
* Tested on: Windows 10, Intel Xeon @ 3.7GHz 32GB, GTX 1070 8GB (SIG Lab)

### Introduction
My GPU rasterizer produces real-time renders of complex geometry. Scene data is loaded from gltf file format.

### Algorithm
1. Initialize array of paths (project a ray from camera through each pixel)
2. Compute intersection with ray along its path
3. Stream compaction to remove terminated paths (optional)
4. Shade rays that intersected something using reflect, refract, or diffuse lighting to multiply with the current color of the ray
5. Repeat steps 2-4 until max bounces reached or all paths terminated
6. Add iteration results to the image, repeating steps 1-5 until max iterations reached

### Images


### Analysis
