# Face Morpher
Applying Delaunay triangulation and affline warping to face morph two pictures.

![Mathurkuriakose](http://imgur.com/NTxSicO.gif)

## Algorithm
The algorithm uses Delaunay's triangulation. To do this, we indiviually select points that match each other on both images. After doing this, triangles are automatically created such that the size of both don't exceed 45 degrees. This allows each individual triangle to be affline warped to the set of points that are changed. After we create our triangulation, we affline warp a certain amount, depending on how far we want our image to look like

## Running Code
```
python main.py <image1> <image2>
```
Note you'll need a dependency on cv2 for the some of the helper functions. Both images also need to be of size 500 x 500 and the same size.
