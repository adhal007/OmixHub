import unittest
from src.bubble_sort import sorter

class BubbleSort(unittest.TestCase):
    def test_sort(self):
        input = [5, 4, 3, 2, 1, 0]
        output = sorter(input)
        self.assertEqual(output, [0, 1, 2, 3, 4, 5])
        input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        output = sorter(input)
        self.assertEqual(output, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        input = list(reversed(range(100)))
        output = sorter(input)
        self.assertEqual(output, list(range(100)))