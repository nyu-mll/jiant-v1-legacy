import itertools
from multiprocessing import Process
import os
import signal
from time import sleep
import tempfile
import unittest

from jiant.utils import serialize


class TestFileLocking(unittest.TestCase):
    def test_normal_writing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Process(
                target=serialize.write_records, args=([x for x in range(10)], tmp_dir + "/tmp.txt")
            )
            p.start()
            p.join()

    def test_write_process_kill_9_does_not_deadlock_next_write(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # the writer will start writing an infinite stream of records to a file
            p1 = Process(
                target=serialize.write_records,
                args=(itertools.count(start=0, step=1), tmp_dir + "/tmp.txt"),
            )
            p1.start()
            # verify that writer created the output file and a lock file
            while not os.path.exists(tmp_dir + "/tmp.txt.lock"):
                sleep(0.01)
            while not os.path.exists(tmp_dir + "/tmp.txt"):
                sleep(0.01)
            # kill 9 writer (simulate an interrupt where lock cleanup is not possible)
            os.kill(p1.pid, signal.SIGKILL)
            # verify the lock for the aborted write still exists
            assert os.path.exists(tmp_dir + "/tmp.txt.lock")
            # verify that a new process can write to the same path as the aborted write
            p2 = Process(
                target=serialize.write_records, args=([x for x in range(10)], tmp_dir + "/tmp.txt")
            )
            p2.start()
            p2.join()

    def test_file_overwriting_is_allowed_when_access_is_not_concurrent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            p1 = Process(
                target=serialize.write_records, args=([x for x in range(10)], tmp_dir + "/tmp.txt")
            )
            p1.start()
            p1.join()
            p2 = Process(
                target=serialize.write_records, args=([x for x in range(10)], tmp_dir + "/tmp.txt")
            )
            p2.start()
            p2.join()
