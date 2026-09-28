# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Tuple
from dataset.bc_dataset import BCReturn
from constants import RunContext, AgentState


class BaseContextFactory(ABC):
    """
    Abstract contract for initializing generation contexts.
    All custom agents must implement this factory.
    """
    @abstractmethod
    async def create(self, question: BCReturn, rollout_idx: int) -> Tuple[RunContext, AgentState]:
        """
        Creates and returns the Context and Start State for a single rollout.
        
        Returns:
            Tuple[Context, StartState]
        """
        pass
