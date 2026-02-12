"""
Test Cases for Island Counter
Author: Oleksandr
"""

import unittest
from counting_islands import IslandCounter, count_islands_from_input


class TestIslandCounter(unittest.TestCase):
    """Unit tests for the IslandCounter class."""
    
    def test_case_1(self):
        """Test case 1: Two separate islands"""
        input_str = """3 3
0 1 0
0 0 0
0 1 1"""
        result = count_islands_from_input(input_str)
        self.assertEqual(result, 2)
    
    def test_case_2(self):
        """Test case 2: Three diagonal islands"""
        input_str = """3 4
0 0 0 1
0 0 1 0
0 1 0 0"""
        result = count_islands_from_input(input_str)
        self.assertEqual(result, 3)
    
    def test_case_3(self):
        """Test case 3: Two connected islands"""
        input_str = """3 4
0 0 0 1
0 0 1 1
0 1 0 1"""
        result = count_islands_from_input(input_str)
        self.assertEqual(result, 2)
    
    def test_empty_matrix(self):
        """Test with empty matrix"""
        matrix = []
        counter = IslandCounter(matrix)
        self.assertEqual(counter.count_islands(), 0)
    
    def test_all_ocean(self):
        """Test matrix with no islands"""
        matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        counter = IslandCounter(matrix)
        self.assertEqual(counter.count_islands(), 0)
    
    def test_all_land(self):
        """Test matrix with one big island"""
        matrix = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        counter = IslandCounter(matrix)
        self.assertEqual(counter.count_islands(), 1)
    
    def test_single_cell_island(self):
        """Test single cell island"""
        matrix = [[1]]
        counter = IslandCounter(matrix)
        self.assertEqual(counter.count_islands(), 1)
    
    def test_large_island(self):
        """Test larger matrix with multiple islands"""
        matrix = [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1]
        ]
        counter = IslandCounter(matrix)
        self.assertEqual(counter.count_islands(), 3)


if __name__ == '__main__':
    unittest.main()
