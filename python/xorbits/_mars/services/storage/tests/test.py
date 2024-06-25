import os
import sys
import asyncio

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow.feather import write_feather, read_feather

from xoscar.api import (
    Actor,
    ActorRef,
    actor_ref,
    BufferRef,
    buffer_ref,
    copy_to,
    create_actor_pool,
    create_actor,
)
from xoscar.backends.allocate_strategy import ProcessIndex
from xoscar.utils import lazy_import

cupy = lazy_import("cupy")


class BufferTransferActor(Actor):
    def __init__(self, cpu: bool = True):
        self.cpu = cpu
        self.xp = np if cpu else cupy
        self._buffers = []

    def create_buffer_refs(self, sizes: list[int]) -> list[BufferRef]:
        if self.cpu:
            buffers = [np.zeros(size, dtype="u1").data for size in sizes]
        else:
            assert cupy is not None
            buffers = [cupy.zeros(size, dtype="u1") for size in sizes]
        self._buffers.extend(buffers)
        res = [buffer_ref(self.address, buf) for buf in buffers]
        return res

    def create_arrays_from_buffer_refs(self, buf_refs: list[BufferRef]):
        if self.cpu:
            return [
                np.frombuffer(BufferRef.get_buffer(ref), dtype="u1") for ref in buf_refs
            ]
        else:
            return [BufferRef.get_buffer(ref) for ref in buf_refs]

    async def copy_array(self, ref: ActorRef, sizes):
        def generate_arrays(low, high, sizes):
            return [
                np.random.randint(low, high, dtype="u1", size=size) for size in sizes
            ]

        arrays1 = generate_arrays(3, 12, sizes)
        arrays2 = generate_arrays(6, 23, sizes)

        if self.cpu:
            buffers1 = [a.data for a in arrays1]
            buffers2 = [a.data for a in arrays2]
        else:
            buffers1 = arrays1 = [self.xp.asarray(a) for a in arrays1]
            buffers2 = arrays2 = [self.xp.asarray(a) for a in arrays2]

        ref: BufferTransferActor = await actor_ref(ref)
        buf_refs1 = await ref.create_buffer_refs(sizes)
        buf_refs2 = await ref.create_buffer_refs(sizes)

        await asyncio.gather(copy_to(buffers1, buf_refs1), copy_to(buffers2, buf_refs2))

        async def verify_arrays(original_arrays, buf_refs):
            new_arrays = await ref.create_arrays_from_buffer_refs(buf_refs)
            assert len(original_arrays) == len(new_arrays)
            for a1, a2 in zip(original_arrays, new_arrays):
                self.xp.testing.assert_array_equal(a1, a2)

        await verify_arrays(arrays1, buf_refs1)
        await verify_arrays(arrays2, buf_refs2)

    async def copy_dataframe(self, ref: ActorRef, sizes: list[int]):
        def generate_dataframes(low, high, sizes):
            return [
                pd.DataFrame(np.random.randint(low, high, size=(rows, cols)))
                for rows, cols in sizes
            ]

        dfs = generate_dataframes(3, 12, sizes)

        buffers = [self._serialize_dataframe(df) for df in dfs]
        buf_sizes = [len(buffer) for buffer in buffers]

        ref: BufferTransferActor = await actor_ref(ref)
        buf_refs = await ref.create_buffer_refs(buf_sizes)

        await copy_to(buffers, buf_refs)

        received_dfs = [
            self._deserialize_dataframe(BufferRef.get_buffer(buf_ref))
            for buf_ref in buf_refs
        ]

        for df, received_df in zip(dfs, received_dfs):
            pd.testing.assert_frame_equal(df, received_df)
            print("DataFrame transmission successful")

    def _serialize_dataframe(self, df: pd.DataFrame) -> bytes:
        output_stream = pa.BufferOutputStream()
        write_feather(df, output_stream)
        return output_stream.getvalue().to_pybytes()

    def _deserialize_dataframe(self, buffer: bytes) -> pd.DataFrame:
        return read_feather(pa.BufferReader(buffer))


async def _start_pool(schemes):
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )

    pool = await create_actor_pool(
        "127.0.0.1",
        n_process=2,
        subprocess_start_method=start_method,
        external_address_schemes=[None, schemes[0], schemes[1]],
    )

    return pool


sizes = [
    [
        10 * 1024**2,
        3 * 1024**2,
        5 * 1024**2,
        8 * 1024**2,
        7 * 1024**2,
    ],
    [
        1 * 1024**2,
        2 * 1024**2,
        1 * 1024**2,
    ],
]

df_sizes = [
    [
        (10 * 1024, 3 * 1024),
        (5 * 1024, 7 * 1024),
        (8 * 1024, 7 * 1024),
    ]
]

schemes = [(None, None), ("ucx", "ucx"), (None, "ucx"), ("ucx", None)]


async def test_dataframe():
    cpu = True
    size = df_sizes[0]
    pool1 = await _start_pool(schemes[0])
    pool2 = await _start_pool(schemes[0])

    async with pool1, pool2:
        actor1: BufferTransferActor = await create_actor(
            BufferTransferActor,
            cpu=cpu,
            uid=f"test_{1}",
            address=pool1.external_address,
            allocate_strategy=ProcessIndex(1),
        )
        actor2 = await create_actor(
            BufferTransferActor,
            cpu=cpu,
            uid=f"test_{2}",
            address=pool2.external_address,
            allocate_strategy=ProcessIndex(1),
        )
        tasks = [actor1.copy_dataframe(actor2, size)]
        await asyncio.gather(*tasks)


async def test_array():
    cpu = True
    size = sizes[0]
    pool1 = await _start_pool(schemes[0])
    pool2 = await _start_pool(schemes[0])

    async with pool1, pool2:
        actor1 = await create_actor(
            BufferTransferActor,
            cpu=cpu,
            uid=f"test_{1}",
            address=pool1.external_address,
            allocate_strategy=ProcessIndex(1),
        )
        actor2 = await create_actor(
            BufferTransferActor,
            cpu=cpu,
            uid=f"test_{2}",
            address=pool2.external_address,
            allocate_strategy=ProcessIndex(1),
        )
        tasks = [actor1.copy_array(actor2, size)]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(test_dataframe())
