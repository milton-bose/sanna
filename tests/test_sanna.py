#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_sanna
----------------------------------

Tests for `sanna` module.
"""

import unittest

#from sanna import sanna


class TestSanna(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_theano_installation(self):
        """Test importing theano...
        """
        import theano
        self.assertIsInstance(str(theano.__version__), str)


if __name__ == '__main__':
    import sys
    sys.exit(unittest.main())
