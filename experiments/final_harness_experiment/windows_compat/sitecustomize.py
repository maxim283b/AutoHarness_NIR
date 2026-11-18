"""Windows-only runtime compatibility for official AutoSaddler.

AutoSaddler intentionally materializes component-map candidates read-only and
then removes their temporary directory. POSIX permits unlinking such files;
Windows does not. Wrap shutil.rmtree so only an EACCES/EPERM target has its
write bit restored before the original operation retries.
"""

from __future__ import annotations

import os
import shutil
import stat


_original_rmtree = shutil.rmtree


def _windows_onexc(function, path, excinfo):
    error = excinfo if isinstance(excinfo, BaseException) else excinfo[1]
    if isinstance(error, PermissionError):
        os.chmod(path, stat.S_IWRITE)
        function(path)
        return
    raise error


def _compatible_rmtree(path, *args, **kwargs):
    if os.name == "nt" and "onexc" not in kwargs and "onerror" not in kwargs:
        kwargs["onexc"] = _windows_onexc
    return _original_rmtree(path, *args, **kwargs)


shutil.rmtree = _compatible_rmtree
