# Task 1: Counting Islands

## Problem Description

Given a matrix M×N representing a map with two possible states:
- `1` - island (land)
- `0` - ocean (water)

Calculate the number of islands in the most effective way. Islands are connected horizontally or vertically (not diagonally).

## Solution Approach

The solution uses **Depth-First Search (DFS)** algorithm to efficiently count islands.

### Algorithm:
1. Iterate through each cell in the matrix
2. When an unvisited land cell (`1`) is found, increment the island counter
3. Perform DFS from that cell to mark all connected land cells as visited
4. Continue until all cells are processed

### Complexity:
- **Time Complexity**: O(M × N) - each cell is visited once
- **Space Complexity**: O(M × N) - for the visited matrix and recursion stack in worst case

## Project Structure

```
task1_counting_islands/
├── counting_islands.py   # Main implementation
├── test_cases.py         # Unit tests
└── README.md            # This file
```

## Installation

No external dependencies required. Only Python 3.6+ is needed.

```bash
python --version  # Ensure Python 3.6+
```

## Usage

### Run with Test Cases

```bash
python counting_islands.py
```

This will run all three provided test cases and display results.

### Run Unit Tests

```bash
python -m unittest test_cases.py
```

Or:

```bash
python test_cases.py
```

### Use in Your Code

```python
from counting_islands import IslandCounter

# Define your matrix
matrix = [
    [0, 1, 0],
    [0, 0, 0],
    [0, 1, 1]
]

# Create counter and get result
counter = IslandCounter(matrix)
num_islands = counter.count_islands()
print(f"Number of islands: {num_islands}")
```

### Input Format

```
M N
row1_values
row2_values
...
rowM_values
```

Where:
- `M` = number of rows
- `N` = number of columns
- Each row contains N space-separated integers (0 or 1)

## Test Cases

### Test Case 1
**Input:**
```
3 3
0 1 0
0 0 0
0 1 1
```
**Output:** `2`

**Explanation:** Two separate islands - one single cell at (0,1) and one two-cell island at (2,1)-(2,2)

### Test Case 2
**Input:**
```
3 4
0 0 0 1
0 0 1 0
0 1 0 0
```
**Output:** `3`

**Explanation:** Three separate single-cell islands on the diagonal

### Test Case 3
**Input:**
```
3 4
0 0 0 1
0 0 1 1
0 1 0 1
```
**Output:** `2`

**Explanation:** Two islands - one L-shaped island on the left and one connected island on the right

## Author

Oleksandr  
February 2026
