# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import random
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from typing import AsyncIterator, Any
import logging
from itertools import islice
import json

logger = logging.getLogger(__name__)

class ParquetDataset:
    def __init__(
        self, 
        path: str, 
        batch_size: int = 256, 
        limit: int | None = None,
        shuffle_buffer_size: int = 512
    ) -> None:
        self.path = path
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.parquet_file = pq.ParquetFile(path)
        
        # 1. Determine the actual total rows in the physical file
        self._physical_total = self.parquet_file.metadata.num_rows
        
        # 2. Cap the effective limit
        if limit is not None:
            self.limit = min(limit, self._physical_total)
        else:
            self.limit = self._physical_total
            
        logger.info(
            f"Opened dataset from {path}. "
            f"Effective samples: {self.limit} (Total in file: {self._physical_total}) | "
            f"Shuffle Buffer: {self.shuffle_buffer_size}"
        )

    def __len__(self) -> int:
        return self.limit

    async def iter_rows(self) -> AsyncIterator[dict[str, Any]]:
        yielded_count = 0
        buffer = []
        
        for batch in self.parquet_file.iter_batches(batch_size=self.batch_size):
            df_chunk = batch.to_pandas()
            
            for _, row in df_chunk.iterrows():
                parsed_row = self._parse_row(row)

                if self.shuffle_buffer_size > 0:
                    # Fill the buffer first
                    if len(buffer) < self.shuffle_buffer_size:
                        buffer.append(parsed_row)
                    else:
                        # Buffer is full: pick a random item to yield, replace it with the new row
                        idx = random.randrange(self.shuffle_buffer_size)
                        yield buffer[idx]
                        buffer[idx] = parsed_row
                        yielded_count += 1
                        
                        # Short-circuit if we hit the limit
                        if yielded_count >= self.limit:
                            return 
                else:
                    # No shuffling, yield directly
                    if yielded_count >= self.limit:
                        return
                        
                    yield parsed_row
                    yielded_count += 1
                
                await asyncio.sleep(0)

        # Flush the remaining buffer if the stream is exhausted 
        # but we haven't hit the total limit yet
        if self.shuffle_buffer_size > 0:
            random.shuffle(buffer)
            for item in buffer:
                if yielded_count >= self.limit:
                    return
                yield item
                yielded_count += 1
                await asyncio.sleep(0)

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self.iter_rows()

    def get_by_index(self, index: int) -> dict[str, Any]:
        """
        Warning: Random access is inherently slow in streaming datasets.
        """
        total_rows = len(self)
        assert 0 <= index < total_rows, f"Index {index} out of range"
        
        dataset = ds.dataset(self.path, format="parquet")
        
        # Read only the specific row from disk
        table_slice = dataset.to_table(filter=None) 
        
        row = table_slice.slice(index, 1).to_pandas().iloc[0]
        return self._parse_row(row)

    def _parse_row(self, row):
        raise NotImplementedError("Subclass should implement _parse_row!")


class JsonlDataset:
    def __init__(
        self, 
        path: str, 
        limit: int | None = None,
        shuffle_buffer_size: int = 512,
        total_rows: int | None = None
    ) -> None:
        self.path = path
        self.shuffle_buffer_size = shuffle_buffer_size
        
        # 1. Determine total rows (fast pass if not explicitly provided)
        if total_rows is not None:
            self._physical_total = total_rows
        else:
            with open(path, 'rb') as f:
                self._physical_total = sum(1 for _ in f)
                
        # 2. Cap the effective limit
        self.limit = min(limit, self._physical_total) if limit is not None else self._physical_total
            
        logger.info(
            f"Opened dataset from {path}. "
            f"Effective samples: {self.limit} (Total in file: {self._physical_total}) | "
            f"Shuffle Buffer: {self.shuffle_buffer_size}"
        )

    def __len__(self) -> int:
        return self.limit

    async def iter_rows(self) -> AsyncIterator[dict[str, Any]]:
        yielded_count = 0
        buffer = []
        
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                parsed_row = self._parse_row(json.loads(line))

                if self.shuffle_buffer_size > 0:
                    if len(buffer) < self.shuffle_buffer_size:
                        buffer.append(parsed_row)
                    else:
                        # Buffer is full: pick a random item to yield, replace with new row
                        idx = random.randrange(self.shuffle_buffer_size)
                        yield buffer[idx]
                        buffer[idx] = parsed_row
                        yielded_count += 1
                        
                        if yielded_count >= self.limit:
                            return 
                else:
                    # No shuffling, yield directly
                    if yielded_count >= self.limit:
                        return
                    yield parsed_row
                    yielded_count += 1
                    
                await asyncio.sleep(0)

        # Flush the remaining buffer if the stream is exhausted 
        # but we haven't hit the total limit yet
        if self.shuffle_buffer_size > 0:
            random.shuffle(buffer)
            for item in buffer:
                if yielded_count >= self.limit:
                    return
                yield item
                yielded_count += 1
                await asyncio.sleep(0)

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self.iter_rows()

    def get_by_index(self, index: int) -> dict[str, Any]:
        """
        Warning: Random access is inherently slow in streaming JSONL datasets (O(N)).
        """
        assert 0 <= index < len(self), f"Index {index} out of range"
        
        with open(self.path, 'r', encoding='utf-8') as f:
            # Use islice to efficiently consume the iterator up to the target index
            line = next(islice(f, index, index + 1))
            
        return self._parse_row(json.loads(line))

    def _parse_row(self, row: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Subclass should implement _parse_row!")
