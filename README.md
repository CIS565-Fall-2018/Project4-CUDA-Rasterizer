CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Xiao Zhang
  * [LinkedIn](https://www.linkedin.com/in/xiao-zhang-674bb8148/)
* Tested on: Windows 10, i7-7700K @ 4.20GHz 16.0GB, GTX 1080 15.96GB (my own PC)

Analysis 
======================
* blocksize1d is set to 128 unchanged

* image order is direct light integrator, full light integrator and naive integrator

* rendering time is measured in second

---

## 1. Checkerboard Scene 

### overview

![](img/checkerboard_render.jpg)

### rendering time

![](img/checkerboard.JPG)

### analysis

xxx.

### images

* checkerboard far distance
  
![](img/checkerboard_far.JPG)

* checkerboard mid distance
  
![](img/checkerboard_mid.JPG)

* checkerboard near distance
  
![](img/checkerboard_near.JPG)

---

## 2. Box Scene 

### overview

![](img/box_render.jpg)

### rendering time

![](img/box.JPG)

### analysis

xxx.

### images

* box far distance
  
![](img/box_far.JPG)

* box mid distance
  
![](img/box_mid.JPG)

* box near distance
  
![](img/box_near.JPG)

---

## 3. Flower Scene 

### overview

![](img/flower_render.jpg)

### rendering time

![](img/flower.JPG)

### analysis

xxx.

### images

* flower far distance
  
![](img/flower_far.JPG)

* flower mid distance
  
![](img/flower_mid.JPG)

* flower near distance
  
![](img/flower_near.JPG)

---

## 4. Duck Scene 

### overview

![](img/duck_render.jpg)

### rendering time

![](img/duck.JPG)

### analysis

xxx.

### images

* duck far distance scan line
  
![](img/duck_far_0.JPG)

* duck far distance tile based
  
![](img/duck_far_1.JPG)

* duck mid distance scan line
  
![](img/duck_mid_0.JPG)

* duck mid distance tile based
  
![](img/duck_mid_1.JPG)

* duck near distance scan line
  
![](img/duck_near_0.JPG)

* duck near distance tile based
  
![](img/duck_near_1.JPG)

---

## 5. Cow Scene 

### overview

![](img/cow_render.jpg)

### rendering time

![](img/cow.JPG)

### analysis

xxx.

### images

* cow far distance scan line
  
![](img/cow_far_0.JPG)

* cow far distance tile based
  
![](img/cow_far_1.JPG)

* cow mid distance scan line
  
![](img/cow_mid_0.JPG)

* cow mid distance tile based
  
![](img/cow_mid_1.JPG)

* cow near distance scan line
  
![](img/cow_near_0.JPG)

* cow near distance tile based
  
![](img/cow_near_1.JPG)

---

## 6. Summary

* xxx. 

---

### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
