
# Runnable Examples

## 1. Interactive demo  
- Add and delete boids with mouse clicks
- Visualize KNN and range queries

The interactive demo is a great way to see how fastquadtree works in practice.
You can see how the quadtree subdivides as you add points, and validate the accuracy of the queries visually.
By pressing 1, you can visualize the KNN query for each boid. 

```bash
pip install -r interactive/requirements.txt
python interactive/interactive_v2.py
```

![Interactive_V2_Screenshot](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/interactive_v2_screenshot.png)

## 1.5 Interactive Demo with Rectangles
- Similar to the above demo, but uses rectangles instead of points
- If the rectangles intersect at all with the query area, they will be highlighted in red

If you are creating a game or simulation environment where entities have bounding boxes, you can use the
rectangular quadtree to quickly check which entities are intersecting with another. 

```bash
pip install -r interactive/requirements.txt
python interactive/interactive_v2_rect.py
```

![Interactive_V2_Rect_Screenshot](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/interactive_v2_rect_screenshot.png)

## 2. Ball Pit  
- Spawn balls in a pit with physics-based collisions
- Easily switch between brute force and quadtree collision detection to see the performance difference

The ball pit demo shows how quadtrees offer massive performance improvements for collision detection.
Rectangular queries are used to find potential collisions, and then precise circle-circle collision checks are performed.

```bash
pip install -r interactive/requirements.txt
python interactive/ball_pit.py
```

![Ballpit_Demo_Screenshot](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/ballpit.png)