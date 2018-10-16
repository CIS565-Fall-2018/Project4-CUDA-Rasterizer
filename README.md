CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Ziad Ben Hadj-Alouane
  * [LinkedIn](https://www.linkedin.com/in/ziadbha/), [personal website](https://www.seas.upenn.edu/~ziadb/)
* Tested on: Windows 10, i7-8750H @ 2.20GHz, 16GB, GTX 1060

# Project Goal
<p align="center">
  <img src="https://github.com/ziedbha/Project4-CUDA-Rasterizer/blob/master/imgs/duck.gif"/>
</p>


In this project, I implemented a Path Tracer using the CUDA parallel computing platform. Path Tracing is a computer graphics Monte Carlo method of rendering images of three-dimensional scenes such that the global illumination is faithful to reality. 

# Video Demo
[<img src="https://github.com/ziedbha/Project4-CUDA-Rasterizer/blob/master/imgs/thumb.png">](https://youtu.be/lv1ACI_r01s)

# Rasterizer Intro
<p align="center">
  <img width="470" height="300" src="https://github.com/ziedbha/Project4-CUDA-Rasterizer/blob/master/imgs/raster.jpg"/>
</p>
Rasterization is a fast rendering technique used in computer graphics. In simple terms, after loading in a model in memory, we identify triangles that constitute the geometry, and then test the intersection of screen pixels with these triangles. In the end, we render the closest intersected pixels. The image above showcases which pixels are considered intersected with the loaded triangle.

This project implements a full rasterization pipeline with GLTF assets support:
1. Load in GLTF asset into memory
2. Transform the vertices from model space to pixel space using camera projection
3. Assemble transformed vertices into packages (normals, colors, textures, etc..) with triangulation
4. Perform triangle/pixel intersection (rasterization)
5. Color identified fragments
6. Send the buffer of colored fragments to the display

## Scenes
### Flower
| Depth View | Normals View | Colored View - Lambert |
| ------------- | ----------- | ----------- |
| <p align="center"><img width="250" height="250" src="https://github.com/ziedbha/Project4-CUDA-Rasterizer/blob/master/imgs/flower_z.png"/></p>| <p align="center"><img width="250" height="250" src="https://github.com/ziedbha/Project4-CUDA-Rasterizer/blob/master/imgs/flower_norm.png/"></p> | <p align="center"><img width="250" height="250" src="https://github.com/ziedbha/Project4-CUDA-Rasterizer/blob/master/imgs/flower_col.png/"></p> |

### Duck
| Depth View | Normals View | Textured View |
| ------------- | ----------- | ----------- |
| <p align="center"><img width="250" height="250" src="https://github.com/ziedbha/Project4-CUDA-Rasterizer/blob/master/imgs/duck_z.png"/></p>| <p align="center"><img width="250" height="250" src="https://github.com/ziedbha/Project4-CUDA-Rasterizer/blob/master/imgs/duck_norm.png/"></p> | <p align="center"><img width="250" height="250" src="https://github.com/ziedbha/Project4-CUDA-Rasterizer/blob/master/imgs/duck_col.png/"></p> |

### CesiumTruck
| Depth View | Normals View | Textured View |
| ------------- | ----------- | ----------- |
| <p align="center"><img width="300" height="200" src="https://github.com/ziedbha/Project4-CUDA-Rasterizer/blob/master/imgs/truck_z.png"/></p>| <p align="center"><img width="300" height="200" src="https://github.com/ziedbha/Project4-CUDA-Rasterizer/blob/master/imgs/truck_norm.png/"></p> | <p align="center"><img width="300" height="200" src="https://github.com/ziedbha/Project4-CUDA-Rasterizer/blob/master/imgs/truck_col.png/"></p> |

## Extra Features
### Supersampling - Anti-Aliasing
| Without SSAA | With SSAA | 
| ------------- | ----------- |
| ![](imgs/no_aa.png) | ![](imgs/aa.png) |

Supersampling is type of anti-aliasing in which we render the image at a higher resolution (say 4x), then for the final output we downsample. This means that if for each final pixel we supersampled 4 other pixels, then the final color of a pixel is the average of those 4. This diminishes issues with stair-step like lines that should be smooth instead. Performance-wise, adding SSAA x4 means that you will be doing x4 amounts of work since your buffer is that much bigger. Everything else stays the same in terms of code structure, it really is just working with bigger buffers.

### Perspective Correct Color
| Without Correct Interp | With Correct Interp | 
| ------------- | ----------- |
| ![](imgs/incorr_col.png) | ![](imgs/corr_col.png) |

Linear interpolation of vertex attributes generally leads to wrong results (colors, normals, texture coordinates) because the geometry is distorted in the world and depth should be taken into account. Applying perspective correct interpolation does that (incorporates the z-value into the calculation) such that the values are interpolated correctly. Performance-wise, this has no impact at all since we are just doing 4 extra floating point computation for the interpolation (dividing by z-values).

### Texture Mapping with Bilinear Filtering
| Without Correct Interp | With Correct Interp |  With Bilinear Filtering | 
| ------------- | ----------- | ----------- |
| ![](imgs/incorr_tex.png) | ![](imgs/corr_tex.png) | ![](imgs/bilinear.png) |

Both images above show texture importing onto a quad. The one on the left shows incorrect interpolation of texture coordinates for the same reason as described above. The middle one shows the correct version. The right version shows a tweaked sampling model called bilinear filtering, which smoothes out displayed textures when they are larger or smaller than stored in memory. Performance-wise, bilinear filtering requires 3 additional global memory reads, which slows down the overall texturing pipeline.

## Performance in the Pipeline
<p align="center">
  <img src="https://github.com/ziedbha/Project4-CUDA-Rasterizer/blob/master/imgs/analysis.png"/>
</p>
Performance varied from scene to scene. For scenes with a large number of triangles, rasterization took the most amount of time because of the large amount of checks done in that stage. Framebuffer copying is the same in all stages (since it is the same operation, independent of scene size). Fragment shading was also almost the same, with the exception of the flower scene that did lambert shading instead of texture mapping, and finally vertex assembly was fast in all cases due to low branching (and the fact that we do it per vertex)

# Build Instructions
1. Install [CMake](https://cmake.org/install/)
2. Install [Visual Studio 2015](https://docs.microsoft.com/en-us/visualstudio/welcome-to-visual-studio-2015?view=vs-2015) with C++ compativility
3. Install [CUDA 8](https://developer.nvidia.com/cuda-80-ga2-download-archive) (make sure your GPU supports this)
4. Clone this Repo
5. Create a folder named "build" in the root of the local repo
6. Navigate to the build folder and run "cmake-gui .." in a CLI
7. Configure the build with Visual Studio 14 2015 Win64, then generate the solution
8. Run the solution using Visual Studio 2015
10. Build cis_565_rasterizer and set it as Startup Project
9. Run cis_565_rasterizer with command line arguments: ../gltfs/name_of_gltf_folder/name_of_gltf_file.gltf


### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
