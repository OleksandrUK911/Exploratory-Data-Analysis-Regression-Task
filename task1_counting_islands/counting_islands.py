"""
Counting Islands Solution
Author: Oleksandr
Date: February 2026

This module implements an efficient solution for counting islands in a binary matrix.
Islands are represented by 1s and ocean by 0s. Islands are connected horizontally or vertically.
"""

from typing import List, Tuple


class IslandCounter:
    """
    A class to count islands in a binary matrix using Depth-First Search (DFS).
    
    Time Complexity: O(M * N) where M is rows and N is columns
    Space Complexity: O(M * N) in worst case for recursion stack
    """
    
    def __init__(self, matrix: List[List[int]]):
        """
        Initialize the IslandCounter with a binary matrix.
        
        Args:
            matrix: A 2D list where 1 represents island and 0 represents ocean
        """
        self.matrix = matrix
        self.rows = len(matrix)
        self.cols = len(matrix[0]) if matrix else 0
        self.visited = [[False] * self.cols for _ in range(self.rows)]
    
    def is_valid(self, row: int, col: int) -> bool:
        """
        Check if the given position is valid and unvisited land.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            True if position is valid, unvisited, and is land (1)
        """
        return (0 <= row < self.rows and 
                0 <= col < self.cols and 
                not self.visited[row][col] and 
                self.matrix[row][col] == 1)
    
    def dfs(self, row: int, col: int) -> None:
        """
        Perform depth-first search to mark all connected land cells.
        
        Args:
            row: Starting row index
            col: Starting column index
        """
        # Mark current cell as visited
        self.visited[row][col] = True
        
        # Define 4 directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Explore all 4 directions
        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy
            if self.is_valid(new_row, new_col):
                self.dfs(new_row, new_col)
    
    def count_islands(self) -> int:
        """
        Count the number of islands in the matrix.
        
        Returns:
            Number of islands found
        """
        if not self.matrix or not self.matrix[0]:
            return 0
        
        island_count = 0
        
        # Iterate through each cell in the matrix
        for i in range(self.rows):
            for j in range(self.cols):
                # If we find unvisited land, it's a new island
                if self.matrix[i][j] == 1 and not self.visited[i][j]:
                    self.dfs(i, j)
                    island_count += 1
        
        return island_count


def parse_input(input_str: str) -> Tuple[int, int, List[List[int]]]:
    """
    Parse input string into matrix dimensions and matrix.
    
    Args:
        input_str: Input string with format:
                   M N
                   row1
                   row2
                   ...
    
    Returns:
        Tuple of (M, N, matrix)
    """
    lines = input_str.strip().split('\n')
    m, n = map(int, lines[0].split())
    matrix = []
    
    for i in range(1, m + 1):
        row = list(map(int, lines[i].split()))
        matrix.append(row)
    
    return m, n, matrix


def count_islands_from_input(input_str: str) -> int:
    """
    Count islands from formatted input string.
    
    Args:
        input_str: Input string containing matrix dimensions and data
        
    Returns:
        Number of islands
    """
    m, n, matrix = parse_input(input_str)
    counter = IslandCounter(matrix)
    return counter.count_islands()


if __name__ == "__main__":
    # Test Case 1
    test1 = """3 3
0 1 0
0 0 0
0 1 1"""
    
    print("Test Case 1:")
    print(test1)
    print(f"Output: {count_islands_from_input(test1)}")
    print(f"Expected: 2\n")
    
    # Test Case 2
    test2 = """3 4
0 0 0 1
0 0 1 0
0 1 0 0"""
    
    print("Test Case 2:")
    print(test2)
    print(f"Output: {count_islands_from_input(test2)}")
    print(f"Expected: 3\n")
    
    # Test Case 3
    test3 = """3 4
0 0 0 1
0 0 1 1
0 1 0 1"""
    
    print("Test Case 3:")
    print(test3)
    print(f"Output: {count_islands_from_input(test3)}")
    print(f"Expected: 2\n")
    
    # Interactive mode
    print("-" * 50)
    print("Enter your own test case:")
    print("Format: M N followed by M rows of N integers")
    print("-" * 50)
