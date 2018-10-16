CUDA Rasterizer
===============

[CLICK ME FOR INSTRUCTION OF THIS PROJECT](./INSTRUCTION.md)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Wanru Zhao
  * [LinkedIn](www.linkedin.com/in/wanru-zhao).
* Tested on: Windows 10, Intel(R) Core(TM) i7-8750H CPU@2.2GHz, GTX 1070 with Max-Q Design(Personal Laptop)

### Final
<p align="middle">
  <img src="imgs/duck2.gif" width="400" />
</p>

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

### Results
#### Base pipeline
Albedo | Depth | Normal
:--:|:--:|:--:
![](imgs/duck.2018-10-16_19-55-39z.png) | ![](imgs/duck.2018-10-16_19-51-15z.png) | ![](imgs/duck.2018-10-16_19-53-56z.png)
Albedo | Lambert | Blinn-phong
![](imgs/duck.2018-10-16_19-55-39z.png) | ![](imgs/duck.2018-10-16_19-57-41z.png) | ![](imgs/duck.2018-10-16_19-58-45z.png)

#### Correct color interpolation
Without color correction | With color correction
:--:|:--:
![](imgs/triangle.2018-10-16_21-18-00z.png) | ![](imgs/triangle.2018-10-16_21-17-39z.png) 

#### UV texture mapping with bilinear texture filtering and perspective correct texture coordinates
Original | Perspective correct texture coordinates | Bilinear filtering + Perspective correct texture coordinates
:--:|:--:|:--:
![](imgs/checkerboard.2018-10-16_20-20-45z.png) | ![](imgs/checkerboard.2018-10-16_20-21-27z.png) | ![](imgs/checkerboard.2018-10-16_20-22-12z.png)

#### Additional primitives
Triangle | Points | Lines
:--:|:--:|:--:
![](imgs/cow.2018-10-16_20-04-43z.png) | ![](imgs/cow.2018-10-16_20-09-45z.png) | ![](imgs/cow.2018-10-16_20-12-06z.png)

#### Backface culling
Original | Lines | Lines with backface culling
:--:|:--:|:--:
![](imgs/CesiumMilkTruck.2018-10-16_20-00-56z.png) | ![](imgs/CesiumMilkTruck.2018-10-16_20-18-27z.png) | ![](imgs/CesiumMilkTruck.2018-10-16_20-19-08z.png)

### Performance Analysis
#### Breakdown of time spent in each pipeline stage
<p align="middle">
  <img src="imgs/Breakdown.png" width="600" />
</p>

Above chart shows that the majority of time spent in rendering pipeline is the stage "Rasterize", where primitives are scaned, overlapped fragments are calculated/stored, and depth test is done to decide the final fragment to be rendered. To avoid race condition, Atomic is used, which takes a considerate amount of time. Generally, more vertices/primitives a model has, more time will be taken for rasterization. (Cow vs Duck) Also the time spent in Rasterize is also related to the size of primitives of a model, since scanline does the calculation for every pixel in AABB of a primitive. (CesiumMilkTruck vs Duck).

For vertex shader and vertex assembly stages, the time spent is simply related to the number of vertices in a model. However, those two stages are very quick and have no significant influence to total process time.

Fragment shader is related to texturing and coloring, and there are only slight different among all models.

#### Performance of backface culling
<p align="middle">
  <img src="imgs/Performance.png" width="600" />
</p>

Since backface culling ignores all backfaces, it benefits "Rasterize" stage the most and has little effect on other stages.

### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
* Image save function transplanted from Project 3
