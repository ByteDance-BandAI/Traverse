#!/usr/bin/env bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# BrowseComp Chinese (the existing 289-question BCZH/BCCH profile).
export BENCHMARK="bczh"
exec "${SCRIPT_DIR}/run_benchmark.sh"
