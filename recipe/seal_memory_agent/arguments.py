# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
from arguments import add_general_args


def build_args():
    parser = argparse.ArgumentParser("Seal Memory Agent")

    parser = add_general_args(parser)

    args = parser.parse_args()
    return args
