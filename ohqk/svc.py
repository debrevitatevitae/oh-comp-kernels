from sklearn import svm
from functools import wraps
import signal


class _TimeoutError(Exception):
    """Exception raised when a timeout occurs"""
    pass


def _timeout(seconds=10, error_message='Timeout'):
    """Timeout decorator based on https://stackoverflow.com/a/22348885/1497446"""
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise _TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


class TimedSVC(svm.SVC):
    timeout = 10 * 60  # 10 m timeout

    @_timeout(timeout)
    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y, sample_weight)
