#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_random
----------------------------------

Tests For Random number generators
"""

import unittest
import numpy as np
import theano
from theano import tensor as T

from sanna.common import random


class TestRandom(unittest.TestCase):

    def test_numpy_rng(self):
        """Test Numpy random number generator
        """
        rng = random.numpy_rng_instance(2512)
        self.assertTrue(
                np.all(
                    rng.randint(0, 500, size=5) ==
                    np.array([204, 60, 368, 186, 6])
                    )
                )

        rng = random.numpy_rng_instance(rng)
        self.assertTrue(
                np.all(
                    rng.randint(0, 500, size=5) ==
                    np.array([198, 100, 237,  76,  69])
                    )
                )

    def test_theano_rng(self):
        """Test Theano random number generator
        """
        rng = random.theano_rng_instance(2512)

        low = T.iscalar('a')
        high = T.iscalar('b')
        size = T.iscalar('size')

        gen_f = theano.function(
                inputs=[low, high, size],
                outputs=rng.random_integers(
                    size=size, low=low, high=high, ndim=1
                    )
                )

        self.assertTrue(
                np.all(
                    gen_f(0, 500, size=5) ==
                    np.array([196, 318, 107, 191, 175])
                    )
                )
        self.assertTrue(
                np.all(
                    gen_f(0, 500, size=5) ==
                    np.array([456, 18, 145, 295, 8])
                    )
                )

        rng = random.theano_rng_instance(rng)
        # Not recompiling recompiling changes numbers
        # The rng depends on number of time it was compiled
        gen_f = theano.function(
                inputs=[low, high, size],
                outputs=rng.random_integers(
                    size=size, low=low, high=high, ndim=1
                    )
                )

        self.assertTrue(
                np.all(
                    gen_f(0, 500, size=5) ==
                    np.array([114, 249, 33, 32, 21])  # number changed
                    )
                )



if __name__ == '__main__':
    import sys
    sys.exit(unittest.main())
