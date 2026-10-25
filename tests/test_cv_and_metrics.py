import numpy as np

from src.features.sample import assign_blocks
from src.models.evaluate import area_for_capture, success_rate_curve


def test_blocks_group_nearby_points():
    x = np.array([100, 200, 15000, 15100])
    y = np.array([100, 150, 100, 120])
    b = assign_blocks(x, y, 10000)
    assert b[0] == b[1] and b[2] == b[3] and b[0] != b[2]


def test_success_rate_perfect_model():
    prob = np.zeros((10, 10))
    dep = np.zeros((10, 10), bool)
    prob[0, :5] = 1.0
    dep[0, :5] = True
    area, cap = success_rate_curve(prob, dep)
    assert cap[-1] == 1.0
    assert area_for_capture(area, cap, 1.0) <= 0.06   # all deposits within top 5% of area
