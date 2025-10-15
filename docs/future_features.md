# Future Features

Below are a list of features that may be added to future versions of this project. 
If you really want any of these features, please let us know by opening an issue.

If you have any suggestions or would like to contribute, please feel free to open an issue or a pull request.

## Planned Features

### 1. Quadtree serialization

By serializing the quadtree, we can save its state to a file and load it later. This will allow us to persist the quadtree structure and data across sessions. For example, you could pre build a quadtree with all the walls in your video game level, serialize it to a file, and then load it when the game starts. This will heavily reduce the game load time since you won't have to rebuild the quadtree from scratch every time.

### 2. Circle support

Currently, we support points and rectangles in two separate quadtrees.
For example, in the ball-pit demo, we use a point quadtree, but then query a larger area to account for the radius of the balls.
With a circle quadtree, we could directly insert circles and perform circle-circle collision detection.

### 3. KNN with criteria function

Currently, KNN only supports finding the nearest neighbors based on euclidean distance.
By adding a criteria function, we could allow users to define custom criteria for finding neighbors by passing a function that 
takes in a point and returns a score. The KNN algorithm would then use this score to determine the nearest neighbors.

### 4. KNN in rectangle quadtree

Currently, KNN is only supported in the point quadtree. By adding KNN support to the rectangle quadtree, we could allow users to find the nearest rectangles to a given point. This would be to the nearest edge of the rectangle, adding complexity to the algorithm.
However, it will allow for really quick collision detection between a point and a set of rectangles as the point can just do
robust-collision handling with the nearest rectangles.