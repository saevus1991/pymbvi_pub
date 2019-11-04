import unittest
import numpy as np
from pymbvi.util import interp_pwc


class TestVariationalEngine(unittest.TestCase):

    def test_unit_interval(self):
        time_grid = np.array([0.0, 1.0])
        grid_val = np.array([3.0])
        time = np.array([0.7])
        test = interp_pwc(time, time_grid, grid_val)
        self.assertEqual(test, 3.0)
        time = np.array([0.0])
        test = interp_pwc(time, time_grid, grid_val)
        self.assertEqual(test, 3.0)
        time = np.array([1.0])
        test = interp_pwc(time, time_grid, grid_val)
        self.assertEqual(test, 3.0)

    def test_general(self):
        time_grid = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        grid_val = np.array([1.0, 2.0, 4.0, 6.0, 7.0])
        time = np.array([0.11, 0.19, 0.31, 0.41, 0.51, 0.61, 0.71, 0.77, 0.91])
        res = np.array([1.0, 1.0, 2.0, 4.0, 4.0, 6.0, 6.0, 6.0, 7.0])
        for i in range(len(time)):
            self.assertEqual(res[i], interp_pwc(time[i], time_grid, grid_val))

        


if __name__ == '__main__':
    unittest.main()