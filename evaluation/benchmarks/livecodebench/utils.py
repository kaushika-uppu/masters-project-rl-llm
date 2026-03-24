from typing import List
import json
import base64
import zlib
import pickle

def parse_public_tests(raw: str) -> dict:
    # public tests are a JSON array of test objects
    tests = json.loads(raw)
    # convert from list of dicts to dict of lists
    return {
        "input": [t["input"] for t in tests],
        "output": [t["output"] for t in tests]
    }

def parse_private_tests(raw: str, n: int) -> dict:
    decoded = base64.b64decode(raw)
    decompressed = zlib.decompress(decoded)
    unpickled = pickle.loads(decompressed)
    tests = json.loads(unpickled)
    return {
        "input": [t["input"] for t in tests][:n],
        "output": [t["output"] for t in tests][:n]
    }

def format_examples(public_tests: dict) -> str:
    examples = ""
    for i, (inp, out) in enumerate(zip(public_tests["input"], public_tests["output"])):
        examples += f"Example {i+1}:\nInput:\n{inp}\nOutput:\n{out}\n"
    return examples.strip()

def evaluate_problem(solution_code: str, tests: dict) -> List[bool]:
    results = []
    for inp, expected in zip(tests["input"], tests["output"]):
        check_program = f"""
{BASE_IMPORTS}
import sys
from io import StringIO

sys.stdin = StringIO('''{inp}''')
output = StringIO()
sys.stdout = output

{solution_code}
actual = output.getvalue().strip()
expected = '''{expected.strip()}'''

assert actual == expected
        """
        result = check_correctness(check_program, timeout=3)
        results.append(result)
    return results

# ===============================================================================================================
# Below is the code for evaluating the generated code. It is copied from the livecodebench dataset utils with slight modifications
# to support a windows computer
# The path is: lcb_runner/evaluation/utils_execute.py
# ===============================================================================================================


# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is adapted from OpenAI's release
# https://github.com/openai/human-eval/blob/master/human_eval/execution.py

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile


BASE_IMPORTS = """from itertools import accumulate, chain, combinations, count, permutations, product, groupby, islice, repeat
from copy import deepcopy
from string import ascii_lowercase
from math import floor, log2, log10, sqrt, comb, gcd, ceil, inf, isqrt
from collections import defaultdict, deque, Counter
from bisect import bisect, bisect_left, bisect_right, insort
from heapq import heappush, heappop, heapify, merge
from functools import reduce, cache, lru_cache
from random import randrange, shuffle
from operator import itemgetter, sub
from re import search as re_search  # Assuming 're' refers to a regex search
from os.path import commonprefix
from typing import List, Tuple, Dict, Set, Optional, Union, Any, Callable, Iterable, Iterator, Generator
import copy
import string
import math
import collections
import bisect
import heapq
import functools
import random
import itertools
import operator
import re
import numpy as np
import pandas as pd
from math import log, prod  # 'log' and 'prod' are functions in the math module
from collections import deque, defaultdict, Counter, OrderedDict
from itertools import accumulate, permutations, combinations, product, groupby, islice, chain, repeat, zip_longest, cycle
from functools import lru_cache, reduce, partial
# from sortedcontainers import SortedList, SortedDict, SortedSet
# import sortedcontainers
from operator import iand
import sys
"""
_manager = None

def get_manager():
    global _manager
    if _manager is None:
        _manager = multiprocessing.Manager()
    return _manager

def check_correctness(check_program, timeout=3):
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """
    manager = get_manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute, args=(check_program, result, timeout))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return result[0] == "passed"


def unsafe_execute(check_program, result, timeout):

    with create_tempdir():

        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()

        # Run program.
        try:
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(check_program, exec_globals)
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


@contextlib.contextmanager
def time_limit(seconds):
    # signal.setitimer is not available on Windows
    if platform.system() == "Windows":
        # On Windows, just yield without timeout (multiprocessing handles it)
        yield
    else:
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")

        signal.setitimer(signal.ITIMER_REAL, seconds)
        signal.signal(signal.SIGALRM, signal_handler)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise OSError

    def readline(self, *args, **kwargs):
        raise OSError

    def readlines(self, *args, **kwargs):
        raise OSError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None