from fastquadtree import QuadTreeObjects


class Person:
    def __init__(self, x: float, y: float, name: str):
        self.x = x
        self.y = y
        self.name = name


qt = QuadTreeObjects((0, 0, 1000, 1000), 16)

people = [
    Person(100, 200, "Alice"),
    Person(150, 250, "Bob"),
    Person(300, 400, "Charlie"),
]

for person in people:
    qt.insert((person.x, person.y), obj=person)

# Find people in a specific area
results = qt.query((90, 190, 200, 300))
for item in results:
    print(f"Found {item.obj.name} at ({item.x}, {item.y})")


# Output:
# Found Alice at (100, 200)
# Found Bob at (150, 250)
