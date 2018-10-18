CUDA Rasterizer
===============

* Salaar Kohari
  * LinkedIn ([https://www.linkedin.com/in/salaarkohari](https://www.linkedin.com/in/salaarkohari))
  * Website ([http://salaar.kohari.com](http://salaar.kohari.com))
  * University of Pennsylvania, CIS 565: GPU Programming and Architecture
* Tested on: Windows 10, Intel Xeon @ 3.7GHz 32GB, GTX 1070 8GB (SIG Lab)

![SSAA][img/Rasterizer.gif]

### Introduction
My GPU rasterizer produces real-time, interactive renders of complex geometry. Scene data is loaded from gltf file format and rendered using the pipeline outlined below.

### Pipeline
1. Vertex shading
2. Primitive assembly with support for triangles read from buffers of index and vertex data
3. Scanline rasterization
4. Fragment shading with lighting scheme
5. Depth buffer for storing and depth testing fragments
6. Fragment-to-depth-buffer writing (with atomics for race avoidance)

### Analysis
![SSAA][img/AA-Visual.png]
![SSAA FPS][img/AA-FPS.png]
Super-sample anti-aliasing renders a higher resolution image and then samples groups pixels when it reduces it down to output resolution. The technique definitely comes at a performance cost as shown in this chart. An optimization of the technique was sampling before transferring from fragment to frame buffer, allowing for a smaller frame buffer and less data passed to the "sendImageToPBO" kernel. This may be further optimized by passing less data into the fragment buffer.

![SSAA FPS][img/Bilinear.png]
Another feature is bilinear UV interpolation in order to reduce artifacts and increase texture quality. This comes at a very small performance cost as shown in the chart. The algorithm was written with minimal computation other than the interpolation (only multiplying at the end, few local variables) in order to be optimized. Shared memory may speed up the memory access part of the procedure, though the low performance cost may not merit further optimization. Vertex color interpolation is also a part of the rasterizer, which comes at a negligible performance cost, since it just requires color to be stored and interpolated once using barycentric coordinates.
