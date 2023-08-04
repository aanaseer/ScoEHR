"""Utility function to create a directory."""

import os


def create(*args):
    # Code from https://github.com/CW-Huang/sdeflow-light/blob/524650bc5ad69522b3e0905672deef0650374512/lib/helpers.py#L119 # noqa: E501
    path = "/".join(a for a in args)
    if not os.path.isdir(path):
        os.makedirs(path)
