CUDA Rasterizer
===============

[CLICK ME FOR INSTRUCTION OF THIS PROJECT](./INSTRUCTION.md)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Siyu Zheng
* Tested on: Windows 10, i7-8750 @ 2.20GHz 16GB, GTX 1060 6GB, Visual Studio 2015, CUDA 8.0(Personal Laptop)

## CUDA Rasterizer

### Shading Methods
| Lambert        | Blinn-Phong            |
| ------------- |:-------------:|
| ![](images/lambert.png)      | ![](images/blinn.png)   |

Use different shading methods to render the object.

### Perspective correct texture 

| Non-perspective correct        | Perspective correct             |
| ------------- |:-------------:|
| ![](images/nonperspective.gif)      | ![](images/perspective.gif)   |

### Bilinear texture filtering
| Non-bilinear       | Bilinear  texture filtering            |
| ------------- |:-------------:|
| ![](images/nobilinear.png)      | ![](images/bilinear.png)   |

### Color Interpolation
| Triangle       | Box            |
| ------------- |:-------------:|
| ![](images/triangle_color.png)      | ![](images/box_color.png)   |

Assigned red, green and blue for vertices of each triangle then use barycentric interpolation to calculate the color of each pixel in that triangle.

### Rasterize point and line
| Point cloud       | Wireframe            |
| ------------- |:-------------:|
| ![](images/point.gif)      | ![](images/line.gif)   |

For rasterizing points, just assign each fragment with a color on that pixel and assign the color to that pixel on frame buffer.

For rasterizing lines, loop all edges for each triangle and calculate the length for each line segment. Divide the line segment into tiny part then assigned color on that pixel. 

### Perform Analysis

![](images/time.png)

Recorded the running time for each process. In general, vertex transform and primitive assembly took about similar amount of time for different test files. We can see that in truck example, the rasterization took most of the time since Cesium Milk Truck has several different texture. 

### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
