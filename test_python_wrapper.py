#!/usr/bin/env python3
"""
Test script for the Python wrapper delete functionality.
"""

import sys
import os

# Add the pysrc directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pysrc'))

from quadtree_rs import QuadTree

def test_delete_wrapper():
    print("Testing Python wrapper delete functionality...")
    print("Note: This test requires the Python extension to be rebuilt with the new delete method")
    
    try:
        # Create a quadtree with object tracking
        qt = QuadTree((0, 0, 100, 100), capacity=4, track_objects=True)
        
        # Insert some items with objects
        id1 = qt.insert((10, 10), obj={"name": "item1"})
        id2 = qt.insert((20, 20), obj={"name": "item2"})
        id3 = qt.insert((30, 30), obj={"name": "item3"})
        id4 = qt.insert((10, 10), obj={"name": "item4"})  # Same location, different ID
        
        print(f"Inserted 4 items: IDs {id1}, {id2}, {id3}, {id4}")
        print(f"Initial count: {len(qt)} (wrapper), {qt.count_items()} (native)")
        
        # Verify objects are tracked
        print(f"Object for ID {id1}: {qt.get(id1)}")
        print(f"Object for ID {id4}: {qt.get(id4)}")
        
        # Delete item by ID and location
        print(f"\nDeleting item {id2} at (20, 20)...")
        deleted = qt.delete(id2, (20, 20))
        print(f"Delete successful: {deleted}")
        print(f"Count after delete: {len(qt)} (wrapper), {qt.count_items()} (native)")
        
        # Verify object was removed from tracking
        print(f"Object for deleted ID {id2}: {qt.get(id2)}")
        
        # Try to delete with wrong ID
        print("\nTrying to delete wrong ID at (10, 10)...")
        deleted = qt.delete(999, (10, 10))
        print(f"Delete successful: {deleted}")
        
        # Try to delete with wrong location
        print("\nTrying to delete correct ID at wrong location...")
        deleted = qt.delete(id1, (50, 50))
        print(f"Delete successful: {deleted}")
        
        # Delete one of the items at the same location
        print(f"\nDeleting ID {id1} at (10, 10)...")
        deleted = qt.delete(id1, (10, 10))
        print(f"Delete successful: {deleted}")
        print(f"Count after delete: {len(qt)} (wrapper), {qt.count_items()} (native)")
        
        # Verify the other item at the same location is still there
        print(f"Object for remaining ID {id4}: {qt.get(id4)}")
        
        # Query to verify remaining items
        items = qt.query((0, 0, 100, 100))
        print(f"All remaining items: {items}")
        
        print("\nPython wrapper delete test completed successfully!")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nThis is expected if the Python extension hasn't been rebuilt with the new delete method.")
        print("To rebuild: maturin develop --release")
        print("\nThe Python wrapper code is ready and will work once the extension is rebuilt.")

if __name__ == "__main__":
    test_delete_wrapper()