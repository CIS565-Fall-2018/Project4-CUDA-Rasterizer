CUDA Rasterizer
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Edward Atter
  * [LinkedIn](https://www.linkedin.com/in/atter/)
  * Tested on: Linux Mint 18.3 Sylvia (4.13.0-41-generic), Ryzen 7 2700x @ 3.7 ghz (base clock) 16GB, GTX 1070 TI 8GB GDDR5 (Personal)
  * CUDA 9

![](img/intro.png)

## Overview

A rasterizer takes 3D objects (defined using vector graphics) and converts it into a 2D representation to be viewed on a computer monitor. Rasterization is the technology behind all major video games as it is much faster than path tracing. 

![](img/box-demo.gif)

_Note:_ The screen door affect oberserved is due to the conversion from `mp4` to `gif` and **not** an affect of the rasterizer.

This project implements a a rasterizer with the following features:
 - Vertex shading
 - Primitive assembly support
 - Rasterization
 - Fragment shading
 - Depth buffer for storing depth testing fragments
 - Fragment-to-depth buffer with mutex lock to prevent race conditions
 - Lambert fragment shader
 - Supersampling Anti-Aliasing (SSAA) using the ["random" algorithm](https://en.wikipedia.org/wiki/Supersampling#Supersampling_patterns)
 - Point cloud drawing mode
 - Line drawing mode

All features may be toggled by changing the defined constants at the start of `src/rasterize.cu`.

## Features

#### Lambert shading

Lambert shading has no noticable affect on performance. The image on the left is without shading, while the image on the right has Lambert shading enabled. The left image has simple normal coloring while the image on the right uses Lambert shading. 

![](img/lambert-comparison.png)


#### Anti-Aliasing via Supersampling

This technique requires upscaling the image, using one of several methods to "average out" the pixels, and setting this new value in the final image. Specifically, this implementation uses the "random" algorithm. 

![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Supersampling_-_Random.svg/180px-Supersampling_-_Random.svg.png)

A list of nearby pixels is generated (the size of which depends on `OPTION_SSAA_GRID_SIZE`). Then multiple random samples are taken from the nearby pixels. The average of the samples represents the final color in the image. 

SSAA has an significant, negative impact on performance. This is unsuprising, since a much larger image must be generated first, typically with four times as many pixels. However, images can look much more realistic. Humans are accustomed to seeing objects in a continuous space in real life, not broken down into discrete pixels. Anti-aliasing aids to more accurately represent a continuous space. 

![](img/ssaa-comparison.png)

#### Alternate Drawing Modes

In addition to the typical `triangle` mode, `line` and `point` modes are also available. If enabled, the original vector graphic is reduced to solely its vertices (point mode). Line mode takes this one step further, connecting each vertex without filling triangle. Triangle mode uses the normal value for coloring, while all other modes are statically defined to be white. Without the need to fill the image, significantly fewer pixels need to be processed. This results in both point and line modes achieving much higher FPS than the normal triangle mode.

![](img/points-cow.png) ![](img/lines-cow.png)

TODO: GRAPHS

## Resources
 - [OpenGL.org, Lambert lighting](https://www.opengl.org/sdk/docs/tutorials/ClockworkCoders/lighting.php)
 - [University of Pennsylvania, CIS 565, Skeleton Project](https://github.com/CIS565-Fall-2018/Project4-CUDA-Rasterizer)
 - [Wikipedia, Supersampling](https://en.wikipedia.org/wiki/Supersampling#Supersampling_patterns)

