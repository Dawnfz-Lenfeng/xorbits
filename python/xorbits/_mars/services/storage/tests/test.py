import os
import sys
import asyncio

import numpy as np
import pytest

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
from xoscar.backends.indigen.pool import MainActorPool
from xoscar.backends.allocate_strategy import ProcessIndex
from xoscar.utils import lazy_import

cupy = lazy_import("cupy")


class BufferTransferActor(Actor):
    def __init__(self, cpu: bool = True):
        self.cpu = cpu
        self.xp = np if cpu else cupy

    def create_buffer_refs(self, sizes: list[int]) -> list[BufferRef]:
        return [self._create_buffer_ref(size) for size in sizes]

    def create_arrays_from_buffer_refs(self, buf_refs: list[BufferRef]):
        if self.cpu:
            return [
                np.frombuffer(BufferRef.get_buffer(ref), dtype="u1") for ref in buf_refs
            ]
        else:
            return [BufferRef.get_buffer(ref) for ref in buf_refs]

    async def copy_data(self, ref: ActorRef, sizes):
        def generate_arrays(low, high):
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

    def _create_buffer_ref(self, size):
        buffer = self.xp.zeros(size, dtype="u1").data
        ref = buffer_ref(self.address, buffer)
        return ref


async def _start_pool(schemes):
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )

    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        subprocess_start_method=start_method,
        external_address_schemes=[None, schemes[0], schemes[1]],
    )

    await pool.start()
    return pool


@pytest.fixture
async def actors(request: pytest.FixtureRequest):
    pool_counts, actor_counts, schemes, cpu = request.param

    pools = []
    for _ in range(pool_counts):
        pools.append(await _start_pool(schemes))

    actors = []
    for pool in pools:
        for i in range(actor_counts):
            actor = await create_actor(
                BufferTransferActor,
                cpu=cpu,
                uid=f"test_{i}",
                address=pool.external_address,
                allocate_strategy=ProcessIndex(i),
            )
            actors.append(actor)

    yield actors

    for pool in pools:
        await pool.stop()


def _generate_params(pool_counts=2, actor_counts=2, cpu=True):
    schemes = [(None, None), ("ucx", "ucx"), (None, "ucx"), ("ucx", None)]

    for scheme in schemes:
        yield pool_counts, actor_counts, scheme, cpu


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


@pytest.mark.asyncio
@pytest.mark.parametrize("actors", _generate_params(), indirect=True)
@pytest.mark.parametrize("sizes", sizes)
async def test_simple_transfer(actors, sizes):
    actor_test = actors[0]
    tasks = [actor_test.copy_data(actor, sizes) for actor in actors[1:]]
    await asyncio.gather(*tasks)
