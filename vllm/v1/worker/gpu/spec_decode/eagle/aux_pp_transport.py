# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct transport for EAGLE3-style aux hidden states under pipeline parallelism.

Drafting runs on the last PP rank but taps layers that may live on earlier
stages. Stages at least two hops from the end send their taps straight to the
last rank; the stage right before it keeps its taps in the
``IntermediateTensors`` handoff it already sends.

Communication lives here rather than in the model forward, which keeps the
forward free of host-side comm state and capturable by full CUDA graphs. The
sender stages a private copy because the forward's output is a capture buffer
the next replay rewrites, while the receiver can write straight into the
persistent input slots the graphs captured.

Send and recv agree on element counts without a metadata exchange because every
rank pads a batch to the same token count. An NCCL count mismatch deadlocks
rather than erroring, so that invariant is load-bearing.

The taps travel on their own communicator, and the connections carrying them
are established at load time. See _new_aux_pp_group and
_warm_up_tap_connections for why skipping either deadlocks from pp_size 3 up.
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from vllm.distributed.parallel_state import get_pp_group, get_world_group
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import _inner_decoder


def _new_aux_pp_group() -> dist.ProcessGroup:
    """Build a second communicator over this rank's PP ranks, for taps only.

    NCCL matches point-to-point operations per communicator, in issue order, so
    a tap sent on the PP communicator sits behind that stage's handoff send. At
    pp_size >= 3 that is a deadlock rather than a slowdown: stage 0 blocks in
    the tap send, stage 1 waits for stage 0's handoff, and the last rank waits
    for stage 1's handoff, so it never reaches the matching tap recv. Giving the
    taps their own communicator removes the ordering relationship.

    Creating a group is collective over the whole world, so every rank walks the
    same sorted list of PP groups and keeps the one it belongs to.
    """
    world = get_world_group()
    mine = list(get_pp_group().ranks)
    gathered: list[list[int] | None] = [None] * world.world_size
    dist.all_gather_object(gathered, mine, group=world.cpu_group)

    my_rank = dist.get_rank()
    aux_group = None
    for ranks in sorted({tuple(r) for r in gathered if r is not None}):
        group = dist.new_group(ranks=list(ranks), backend="nccl")
        if my_rank in ranks:
            aux_group = group
    assert aux_group is not None, "this rank belongs to no PP group"
    return aux_group


def _warm_up_tap_connections(
    group: dist.ProcessGroup, producers: list[int], dst: int
) -> None:
    """Open each producer-to-last connection while every rank is idle.

    isend does not return until the matching recv is posted, because an
    unbatched point-to-point op builds a two-rank NCCL communicator on first
    use and that build is a rendezvous. Measured on 3 ranks: the first isend
    returns only once the peer arrives, 8.4s later, while a second isend on the
    same pair returns in under a millisecond.

    That first send is unserviceable mid-step. The last rank cannot post the
    matching recv until it has taken delivery of the handoff, and the handoff
    is behind the very send that is blocked, so the pipeline closes a cycle on
    itself. Paying the rendezvous here, once, costs two bytes per producer at
    load time and leaves every later send genuinely asynchronous.
    """
    me = dist.get_rank()
    probe = torch.zeros(1, dtype=torch.bfloat16, device="cuda")
    for src in producers:
        # Sequential by producer: only the two peers involved take part, and
        # neither is waiting on anything else at this point.
        if me == src:
            dist.isend(probe, dst=dst, group=group).wait()
        elif me == dst:
            dist.irecv(probe, src=src, group=group).wait()


class AuxTapSender:
    """Sends a far stage's local aux taps straight to the last PP rank."""

    def __init__(self, key_prefix: str, max_in_flight: int, group: dist.ProcessGroup):
        pp = get_pp_group()
        self._group = group
        self._dst = pp.ranks[-1]
        self._key_prefix = key_prefix
        self._max_in_flight = max_in_flight
        # (handle, staging tensor): the reference keeps the staging buffer
        # alive until NCCL has finished reading it.
        self._in_flight: list[tuple[dist.Work, torch.Tensor]] = []

    def _reap(self) -> None:
        """Drop completed sends, bounding how many stay pinned.

        A stage runs ahead of the last rank by up to ``pp_size`` microbatches,
        so waiting unconditionally would stall the pipeline.
        """
        pending = [(h, t) for h, t in self._in_flight if not h.is_completed()]
        while len(pending) > self._max_in_flight:
            handle, _ = pending.pop(0)
            handle.wait()
        self._in_flight = pending

    def extract_and_send(
        self, intermediate_tensors: IntermediateTensors
    ) -> IntermediateTensors:
        """Send the aux taps packed into the forward's output and return the
        handoff payload without them, so they are not also relayed hop-by-hop
        to the next stage."""
        tensors = intermediate_tensors.tensors
        aux_keys = [k for k in tensors if k.startswith(self._key_prefix)]
        if not aux_keys:
            return intermediate_tensors
        aux_keys.sort(key=lambda k: int(k[len(self._key_prefix) :]))

        self._reap()
        for key in aux_keys:
            # A private copy: under full-cudagraph the tap lives in a capture
            # buffer that the next replay rewrites, while this send can stay
            # in flight for several steps.
            staged = tensors[key].clone(memory_format=torch.contiguous_format)
            handle = dist.isend(staged, dst=self._dst, group=self._group)
            self._in_flight.append((handle, staged))
        return IntermediateTensors(
            {k: v for k, v in tensors.items() if not k.startswith(self._key_prefix)}
        )


class AuxTapReceiver:
    """Receives far stages' aux taps into the last rank's persistent buffer."""

    def __init__(self, slots: list[tuple[int, str]], group: dist.ProcessGroup):
        # (source global rank, persistent-buffer key), ordered by producer
        # rank then tap, matching each sender's send order per rank.
        self._group = group
        self._slots = slots

    def recv_into(
        self, intermediate_tensors: IntermediateTensors, num_tokens: int
    ) -> None:
        handles = [
            dist.irecv(
                intermediate_tensors[key][:num_tokens], src=src, group=self._group
            )
            for src, key in self._slots
        ]
        for handle in handles:
            handle.wait()


def init_aux_pp_transport(
    model: nn.Module,
) -> tuple[AuxTapSender | None, AuxTapReceiver | None]:
    """Build this rank's side of the aux transport, if it has one.

    Returns ``(sender, receiver)``. At ``pp_size <= 2`` both are None: the
    only producer is the stage right before the last one, whose taps ride the
    pipeline handoff. The same holds for the pre-last stage at any size.
    """
    pp = get_pp_group()
    if pp.world_size <= 2:
        return None, None
    inner = _inner_decoder(model)
    if inner is None or not getattr(inner, "supports_aux_hidden_states_over_pp", False):
        return None, None

    last = pp.world_size - 1
    rank = pp.rank_in_group
    prefix = inner.AUX_HIDDEN_STATE_KEY

    # The layout follows from the partition, not from who is asking, so every
    # rank derives the same producer list and the setup below stays collective.
    producers = [
        r
        for r in range(last - 1)
        if inner._num_local_taps_on_rank(r, pp.world_size) > 0
    ]
    if not producers:
        return None, None

    aux_group = _new_aux_pp_group()
    _warm_up_tap_connections(
        aux_group, [pp.ranks[r] for r in producers], pp.ranks[last]
    )

    if rank < last - 1:
        if rank not in producers:
            return None, None
        return AuxTapSender(prefix, pp.world_size, aux_group), None
    if rank == last:
        slots: list[tuple[int, str]] = []
        for r in range(last - 1):
            base = inner._aux_slot_base(r, pp.world_size)
            for j in range(inner._num_local_taps_on_rank(r, pp.world_size)):
                slots.append((pp.ranks[r], f"{prefix}{base + j}"))
        if not slots:
            return None, None
        return None, AuxTapReceiver(slots, aux_group)
    # Stage right before the last one: taps ride the handoff.
    return None, None
