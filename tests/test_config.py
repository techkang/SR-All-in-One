import os
import tempfile
import unittest

from config import get_cfg

CFG_STR = """
datasets:
    train: 'Set14'
    test: 'Set14'
"""


class TestConfig(unittest.TestCase):
    def _merge_cfg_str(self, cfg, merge_str):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        try:
            f.write(merge_str)
            f.close()
            cfg.merge_from_file(f.name)
        finally:
            os.remove(f.name)
        return cfg

    def test_merge(self):
        cfg = get_cfg()
        self._merge_cfg_str(cfg, CFG_STR)
        self.assertEqual(cfg.datasets.train, 'Set14')
