import unittest

import torch as t
from model.dcnn import DCNNDenoiser


class TestDCNNDenoiser(unittest.TestCase):
    def setUp(self) -> None:
        self.net = DCNNDenoiser()
        self.dummy_input = t.randn((4, 1, 32, 32))

    def test_net(self):
        output = self.net(self.dummy_input)
        assert output.shape == self.dummy_input.shape
