#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py
from tests import bilateral_test

import pstats, cProfile
import run_tests

cProfile.runctx("bilateral_test.testit()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
