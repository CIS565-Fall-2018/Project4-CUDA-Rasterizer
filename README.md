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

### Perspective correct texture 

| Non-perspective correct        | Perspective correct             |
| ------------- |:-------------:|
| ![](images/nonperspective.gif)      | ![](images/perspective.gif)   |

### Bilinear texture filtering
| Non-bilinear       | Bilinear  texture filtering            |
| ------------- |:-------------:|
| ![](images/nobilinear.png)      | ![](images/bilinear.png)   |

### Rasterize point and line
| Point cloud       | Wireframe            |
| ------------- |:-------------:|
| ![](images/point.gif)      | ![](images/line.gif)   |

### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
