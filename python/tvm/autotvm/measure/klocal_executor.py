"""Local based implementation of the executor using multiprocessing"""

import signal, os
import tvm
import logging

from multiprocessing import Process, Queue
try:
    from queue import Empty
except ImportError:
    from Queue import Empty

try:
    import psutil
except ImportError:
    psutil = None

from . import executor
from . import local_executor
from .local_executor import LocalExecutor, kill_child_processes


def _kexecute_func(func, queue, target, args, kwargs):
    """execute function and return the result or exception to a queue"""
    ctx = tvm.context(str(target))
    if not ctx.exist:
        raise RuntimeError("Cannot get context from local devices. ",
						"Please check you have a suitable device for target: ", target)
    try:
        res = func(ctx, *args, **kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        res = exc
    queue.put(res)

def kcall_with_timeout(queue, timeout, func, target, args, kwargs):
    """A wrapper to support timeout of a function call"""
    # start a new process for timeout (cannot use thread because we have c function)
    p = Process(target=_kexecute_func, args=(func, queue, target, args, kwargs))

    p.start()
    p.join(timeout=timeout)

    queue.put(executor.TimeoutError())

    kill_child_processes(p.pid)
    p.terminate()
    p.join()


class KLocalFuture(executor.Future):
    """Local wrapper for the future
    Similar to LocalFuture but takes a result queue for init.
    Used for KLocalExecutor where there is only one fork for timeout.

    Parameters
    ----------
    queue: multiprocessing.Queue
        queue that receivied the result of the task
    """
    def __init__(self, queue):
        self._queue = queue
        self.res = None
    def done(self):
        return True

    def get(self, timeout=None):
        if self.res is None:

            try:
                res = self._queue.get(block=True, timeout=timeout)
                self.res = res
            except Empty:
                raise executor.TimeoutError()
            self._queue.close()
            self._queue.join_thread()
            del self._queue
            return res
        else:
            return self.res


class KLocalExecutor(LocalExecutor):
    """Local executor that runs workers on the same machine with multiprocessing
    but the initialisation is done in the measured process and is guaranted that
    only one working process is alive at a time.

    Parameters
    ----------
    timeout: float, optional
        timeout of a job. If time is out. A TimeoutError will be returned (not raised)
    do_fork: bool, optional
        For some runtime systems that do not support fork after initialization
        (e.g. cuda runtime, cudnn). Set this to False if you have used these runtime
        before submitting jobs.
    """

    def submit(self, func, target, *args, **kwargs):
        queue = Queue(2)  # Size of 2 to avoid a race condition with size 1.
        kcall_with_timeout(queue, self.timeout, func, target, args, kwargs)
        return KLocalFuture(queue)
