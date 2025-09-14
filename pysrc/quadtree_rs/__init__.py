# Re-export the class from the compiled submodule next to this file.
# The *.pyd inside this folder is a submodule named "quadtree_rs.quadtree_rs".
from .quadtree_rs import QuadTree

__all__ = ["QuadTree"]
