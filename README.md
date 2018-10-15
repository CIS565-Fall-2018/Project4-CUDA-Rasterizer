CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Lan Lou
	*  [github](https://github.com/LanLou123).
* Tested on: Windows 10, i7-6700HQ @ 2.60GHz 16GB, GTX 1070 8GB (Personal Laptop)

### Sample Rasterization

#### ```resolution```: 900X900 ```GLTF model```: cesiummilktruck ```shader``` : blinn_phong perspective corrected bilinear textureed
![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/truck.gif)


diffuse buffer|depth buffer|
------------|--------
![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/diffuse.gif) | ![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/depth.gif)  

normal buffer|specular buffer|
------------|--------
![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/normal.gif) | ![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/spec.gif)

#### combined:
![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/res.gif)

point|line|triangle
-----|----|-----
![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/p.gif) | ![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/line.gif) | ![](https://github.com/LanLou123/Project4-CUDA-Rasterizer/raw/master/renders/lamm.gif)



### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
