"""Run the regression suite without calling external prediction services."""

import os
from pathlib import Path
import tempfile
import unittest


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="deception-checks-") as temporary:
        # Isolate library caches; do not read or change the user's MNE settings.
        os.environ["MNE_DONTWRITE_HOME"] = "true"
        os.environ["NUMBA_CACHE_DIR"] = str(Path(temporary) / "numba")
        os.environ["MPLCONFIGDIR"] = str(Path(temporary) / "matplotlib")
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
