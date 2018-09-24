from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['GCS_READ_CACHE_DISABLED'] = '1'

import sys

from tensorboard import program


def run_main():
    program.setup_environment()
    tensorboard = program.TensorBoard()
    try:
        from absl import app
        from absl.flags import argparse_flags
        app.run(tensorboard.main, flags_parser=tensorboard.configure)
        raise AssertionError("absl.app.run() shouldn't return")
    except ImportError:
        pass
    tensorboard.configure(sys.argv)
    sys.exit(tensorboard.main())


if __name__ == '__main__':
    run_main()
