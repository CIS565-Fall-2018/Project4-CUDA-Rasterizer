CUDA Rasterizer
===============

[CLICK ME FOR INSTRUCTION OF THIS PROJECT](./INSTRUCTION.md)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Wanru Zhao
  * [LinkedIn](www.linkedin.com/in/wanru-zhao).
* Tested on: Windows 10, Intel(R) Core(TM) i7-8750H CPU@2.2GHz, GTX 1070 with Max-Q Design(Personal Laptop)

### Final

### Features
#### Basic Features
- Vertex shading
- Primitive assembly with different primitive modes
- Rasterization
- A depth buffer for storing and depth testing fragments
- Fragment-to-depth-buffer writing (with atomics for race avoidance)
- Fragment shading (lambert and blinn-phong)
#### Extra Features
- Backface culling
- Correct color interpolation between points on a primitive
- UV texture mapping with bilinear texture filtering and perspective correct texture coordinates
- Support for rasterizing additional primitives: lines and points

### Performance Analysis



### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
