"""Site-specific customization hook.

Fixes a known Python 3.12 Windows issue where platform.system() / platform.uname()
deadlocks on Windows Management Instrumentation (WMI) queries during module imports.
"""

import platform
from collections import namedtuple

try:
    _uname_result = namedtuple(
        "uname_result", ["system", "node", "release", "version", "machine", "processor"]
    )
    _cached_uname = _uname_result("Windows", "OMEN", "10", "10.0.22631", "AMD64", "Intel64")

    platform.system = lambda: "Windows"
    platform.uname = lambda: _cached_uname
except Exception:
    pass
