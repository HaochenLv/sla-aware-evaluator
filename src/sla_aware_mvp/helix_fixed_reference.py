from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .domain import Pipeline, RequestSpec, SLA, sorted_workload


@dataclass(frozen=True)
class HelixQueryMetrics:
    request_id: str
    aligned_ttft_s: float
    true_first_token_ttft_s: float
    max_tpot_s: float
    decode_tpot_s: tuple[float, ...]


@dataclass(frozen=True)
class HelixReferenceResult:
    feasible: bool
    finished_requests: int
    total_requests: int
    final_time_s: float
    first_violation_kind: str | None
    first_violation_request_id: str | None
    first_violation_observed_s: float | None
    first_violation_limit_s: float | None
    query_metrics: Mapping[str, HelixQueryMetrics]


@dataclass(frozen=True)
class _HelixRuntime:
    ClusterSimulator: type
    ModelName: Any
    create_model: Any
    PipelineStage: type
    Query: type
    BaseScheduler: type
    TransmissionSchedule: type
    TransmissionType: Any
    ExecutionSchedule: type
    execution_policy: Any
    TOKEN_SLOW_LINK: float


def _load_helix_runtime(helix_root: str | Path) -> _HelixRuntime:
    root = str(Path(helix_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

    from simulator.event_simulator.cluster_simulator import ClusterSimulator
    from simulator.event_simulator.model import create_model
    from simulator.event_simulator.query_manager import Query
    from simulator.event_simulator.request import PipelineStage
    from simulator.event_simulator.utils import TOKEN_SLOW_LINK
    from simulator.model_manager.model_manager import ModelName
    from simulator.scheduler.base_scheduler import (
        BaseScheduler,
        ExecutionSchedule,
        TransmissionSchedule,
        TransmissionType,
    )
    from simulator.scheduler.execution_policy import execution_policy

    return _HelixRuntime(
        ClusterSimulator=ClusterSimulator,
        ModelName=ModelName,
        create_model=create_model,
        PipelineStage=PipelineStage,
        Query=Query,
        BaseScheduler=BaseScheduler,
        TransmissionSchedule=TransmissionSchedule,
        TransmissionType=TransmissionType,
        ExecutionSchedule=ExecutionSchedule,
        execution_policy=execution_policy,
        TOKEN_SLOW_LINK=TOKEN_SLOW_LINK,
    )


def _helix_machine_type(hardware_type: str) -> str:
    normalized = hardware_type.strip().lower()
    aliases = {
        "a100": "A100",
        "a100-40gb": "A100",
        "nvidia a100": "A100",
        "t4": "T4",
        "l4": "L4",
        "v100": "V100",
    }
    if normalized not in aliases:
        raise ValueError(
            f"HELIX adapter has no machine-type mapping for {hardware_type!r}"
        )
    return aliases[normalized]


def _ordered_stages(pipeline: Pipeline):
    pipeline.validate()
    return tuple(sorted(pipeline.stages, key=lambda stage: stage.layer_start))


def validate_helix_fixed_reference_pipeline(pipeline: Pipeline) -> None:
    """Validate the deliberately narrow first Reference integration.

    The current adapter supports the project's present validation pipelines:
    consecutive layer stages connected by exactly one physical link. It does not
    invent pass-through compute stages for multi-hop physical routes.
    """
    ordered = _ordered_stages(pipeline)
    if len({stage.node_id for stage in ordered}) != len(ordered):
        raise ValueError(
            "HELIX fixed Reference currently requires one contiguous stage per node"
        )
    boundary_by_pair = {
        (boundary.upstream_stage_id, boundary.downstream_stage_id): boundary
        for boundary in pipeline.boundaries
    }
    for upstream, downstream in zip(ordered, ordered[1:]):
        boundary = boundary_by_pair[(upstream.id, downstream.id)]
        if upstream.node_id == downstream.node_id:
            raise ValueError(
                "adjacent same-node stages should be merged before HELIX replay"
            )
        if len(boundary.route_link_ids) != 1:
            raise ValueError(
                "HELIX fixed Reference MVP supports one direct physical link per stage boundary"
            )


def _make_fixed_pipeline_scheduler(runtime: _HelixRuntime):
    """Route every request along its pre-attached mini-pipeline.

    Only routing is fixed here. GPU batching/execution is delegated unchanged to
    HELIX's public execution_policy. Link service uses HELIX's normal transmission
    mechanism and available bandwidth accounting.
    """

    class FixedPipelineScheduler(runtime.BaseScheduler):
        def schedule_transmission(self, node):
            free_links: dict[int, float] = {}
            for link_uid, link in node.outbound_links.items():
                flow_control = (
                    node.outbound_nic_speed / len(node.outbound_links)
                    - (link.bandwidth - link.available_bandwidth)
                )
                available = min(flow_control, link.get_available_bandwidth())
                if available > runtime.TOKEN_SLOW_LINK:
                    free_links[link_uid] = available

            assigned: set[int] = set()
            schedules = []
            for request in node.outbound_request_dict.values():
                next_stage = request.get_next_pipeline_stage()
                link_uid = next_stage.link_uid
                if link_uid not in free_links or link_uid in assigned:
                    continue
                request.march_pipeline_stage()
                assigned.add(link_uid)
                schedules.append(
                    runtime.TransmissionSchedule(
                        link_uid=link_uid,
                        bandwidth_usage=free_links[link_uid],
                        requests=[request],
                        transmission_type=runtime.TransmissionType.NormalExecution,
                    )
                )
            return schedules, []

        def schedule_execution(self, node, executable_requests):
            return runtime.execution_policy(
                node=node, executable_requests=executable_requests
            )

        def schedule_model_loading(self):
            return None

    return FixedPipelineScheduler()


def _build_helix_simulator(
    *, pipeline: Pipeline, runtime: _HelixRuntime
) -> tuple[Any, list[Any], dict[str, int]]:
    validate_helix_fixed_reference_pipeline(pipeline)
    ordered = _ordered_stages(pipeline)
    machine_by_node = {
        stage.node_id: _helix_machine_type(pipeline.nodes[stage.node_id].hardware_type)
        for stage in ordered
    }
    machine_num_dict: dict[str, int] = {}
    for machine in machine_by_node.values():
        machine_num_dict[machine] = machine_num_dict.get(machine, 0) + 1

    simulator = runtime.ClusterSimulator(
        model_name=runtime.ModelName.LLaMa70B,
        machine_num_dict=machine_num_dict,
    )
    if simulator.model_manager.get_num_layers() != pipeline.model.num_layers:
        raise ValueError(
            "pipeline model layer count does not match HELIX LLaMA-70B model"
        )

    active_link_ids = []
    boundary_by_pair = {
        (boundary.upstream_stage_id, boundary.downstream_stage_id): boundary
        for boundary in pipeline.boundaries
    }
    for upstream, downstream in zip(ordered, ordered[1:]):
        active_link_ids.extend(
            boundary_by_pair[(upstream.id, downstream.id)].route_link_ids
        )
    max_link_capacity = max(
        (pipeline.links[link_id].capacity_bytes_per_s for link_id in active_link_ids),
        default=1e9,
    )
    endpoint_bandwidth = max(1e12, max_link_capacity * 1000.0)

    full_model = runtime.create_model(
        layer_parameter_sizes=simulator.model_manager.get_model_params()
    )
    source, sink = simulator.initialize(
        coordinator_inbound_nic_speed=endpoint_bandwidth,
        coordinator_outbound_nic_speed=endpoint_bandwidth,
        full_model=full_model,
        machine_types=sorted(set(machine_by_node.values())),
    )

    helix_node_by_id: dict[str, Any] = {}
    helix_uid_by_node_id: dict[str, int] = {}
    for stage in ordered:
        node = pipeline.nodes[stage.node_id]
        machine = machine_by_node[stage.node_id]
        kv_capacity = simulator.model_manager.get_kv_cache_capacity(
            machine_type=machine, num_on_node_layers=stage.num_layers
        )
        helix_node = simulator.add_compute_node(
            vram_size=float(node.memory_capacity_bytes),
            inbound_nic_speed=endpoint_bandwidth,
            outbound_nic_speed=endpoint_bandwidth,
            disk_speed=1e15,
            machine_type=machine,
            kv_cache_capacity=kv_capacity,
            activation_backup_capacity=0,
        )
        helix_node_by_id[stage.node_id] = helix_node
        helix_uid_by_node_id[stage.node_id] = helix_node.node_uid

    source_link = simulator.add_network_link(
        node_in_uid=source.node_uid,
        node_out_uid=helix_node_by_id[ordered[0].node_id].node_uid,
        latency=0.0,
        bandwidth=endpoint_bandwidth,
    )
    helix_link_uid_by_id: dict[str, int] = {}
    for upstream, downstream in zip(ordered, ordered[1:]):
        boundary = boundary_by_pair[(upstream.id, downstream.id)]
        link_id = boundary.route_link_ids[0]
        link = pipeline.links[link_id]
        helix_link = simulator.add_network_link(
            node_in_uid=helix_node_by_id[upstream.node_id].node_uid,
            node_out_uid=helix_node_by_id[downstream.node_id].node_uid,
            latency=0.0,
            bandwidth=link.capacity_bytes_per_s,
        )
        helix_link_uid_by_id[link_id] = helix_link.link_uid
    sink_link = simulator.add_network_link(
        node_in_uid=helix_node_by_id[ordered[-1].node_id].node_uid,
        node_out_uid=sink.node_uid,
        latency=0.0,
        bandwidth=endpoint_bandwidth,
    )

    simulator.set_scheduler(_make_fixed_pipeline_scheduler(runtime))
    simulator.init_query_manager()
    simulator.mark_as_ready()

    for stage in ordered:
        simulator.issue_command_load_model(
            load_time=0.0,
            node_uid=helix_uid_by_node_id[stage.node_id],
            new_layers=list(range(stage.layer_start, stage.layer_end)),
            request_uids_to_wait=[],
        )

    # Model load uses the HELIX event mechanism. With the intentionally huge disk
    # rate above this is only a handful of events; stop as soon as every stage is loaded.
    load_events = 0
    while any(
        len(helix_node_by_id[stage.node_id].in_vram_model_layers) != stage.num_layers
        for stage in ordered
    ):
        succeeded, _ = simulator.simulate_next_event()
        load_events += 1
        if not succeeded or load_events > 10_000:
            raise RuntimeError("HELIX model loading did not complete")

    mini_pipeline = [
        runtime.PipelineStage(
            link_uid=source_link.link_uid,
            bandwidth_usage=-1,
            node_uid=helix_uid_by_node_id[ordered[0].node_id],
            layers_to_infer=list(
                range(ordered[0].layer_start, ordered[0].layer_end)
            ),
        )
    ]
    for upstream, downstream in zip(ordered, ordered[1:]):
        boundary = boundary_by_pair[(upstream.id, downstream.id)]
        link_id = boundary.route_link_ids[0]
        mini_pipeline.append(
            runtime.PipelineStage(
                link_uid=helix_link_uid_by_id[link_id],
                bandwidth_usage=-1,
                node_uid=helix_uid_by_node_id[downstream.node_id],
                layers_to_infer=list(
                    range(downstream.layer_start, downstream.layer_end)
                ),
            )
        )
    mini_pipeline.append(
        runtime.PipelineStage(
            link_uid=sink_link.link_uid,
            bandwidth_usage=-1,
            node_uid=sink.node_uid,
            layers_to_infer=None,
        )
    )
    return simulator, mini_pipeline, helix_uid_by_node_id


def _issue_fixed_query(
    *,
    simulator: Any,
    runtime: _HelixRuntime,
    creation_time: float,
    input_tokens: int,
    output_tokens: int,
    mini_pipeline: Sequence[Any],
) -> int:
    """Mirror HELIX QueryManager.issue_query, changing only pipeline=None -> fixed pipeline."""
    manager = simulator.query_manager
    query_uid = manager.get_next_query_uid()
    query = runtime.Query(
        query_uid=query_uid,
        creation_time=creation_time,
        input_seq_length=input_tokens,
        output_seq_length=output_tokens,
        total_num_layers=manager.param.total_num_layers,
    )
    manager.queries_on_the_fly[query_uid] = query
    phase, token_seq_length, inferred_token_count = query.get_next_iteration()
    simulator.issue_command_new_request(
        base_query_uid=query_uid,
        arrive_time=creation_time,
        phase=phase,
        token_seq_length=token_seq_length,
        prev_num_tokens=inferred_token_count,
        token_size=manager.param.token_size,
        activation_size=manager.param.token_activation_size,
        pipeline=copy.deepcopy(list(mini_pipeline)),
        kv_tracker_ref=query.kv_tracker,
    )
    return query_uid


def extract_helix_query_metrics(
    *, request_id: str, query: Any, overhead_s: float = 0.0
) -> HelixQueryMetrics:
    history = list(query.inference_history)
    if len(history) < 2:
        raise ValueError("finished HELIX query must contain Prefill and at least one Decode iteration")
    prefill = history[0]
    decode = history[1:]
    aligned_ttft = prefill.end_time - query.creation_time + overhead_s
    true_first_token_ttft = decode[0].end_time - query.creation_time + overhead_s
    decode_tpot = tuple(
        item.end_time - item.start_time + overhead_s for item in decode
    )
    return HelixQueryMetrics(
        request_id=request_id,
        aligned_ttft_s=aligned_ttft,
        true_first_token_ttft_s=true_first_token_ttft,
        max_tpot_s=max(decode_tpot),
        decode_tpot_s=decode_tpot,
    )


def evaluate_helix_fixed_reference(
    *,
    pipeline: Pipeline,
    workload: Sequence[RequestSpec],
    sla: SLA,
    helix_root: str | Path,
    max_events: int = 5_000_000,
) -> HelixReferenceResult:
    """Replay one finite workload on HELIX with an externally fixed pipeline.

    Feasibility is evaluator-aligned for TTFT: Prefill completion is compared with
    the current Conservative evaluator's TTFT budget. True first-token TTFT is also
    returned for later sensitivity checks. TPOT is the end-to-end duration of each
    HELIX Increment iteration. No HELIX placement/max-flow optimization is invoked.
    """
    requests = sorted_workload(workload)
    if not requests:
        return HelixReferenceResult(True, 0, 0, 0.0, None, None, None, None, {})

    runtime = _load_helix_runtime(helix_root)
    simulator, mini_pipeline, _ = _build_helix_simulator(
        pipeline=pipeline, runtime=runtime
    )
    base_time = simulator.current_time
    query_uid_to_request_id: dict[int, str] = {}
    for request in requests:
        query_uid = _issue_fixed_query(
            simulator=simulator,
            runtime=runtime,
            creation_time=base_time + request.arrival_time_s,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            mini_pipeline=mini_pipeline,
        )
        query_uid_to_request_id[query_uid] = request.id

    processed_events = 0
    while simulator.query_manager.queries_on_the_fly:
        succeeded, _ = simulator.simulate_next_event()
        processed_events += 1
        if not succeeded:
            raise RuntimeError("HELIX event queue drained with unfinished queries")
        if processed_events > max_events:
            raise RuntimeError("HELIX fixed Reference event limit exceeded")

    metrics: dict[str, HelixQueryMetrics] = {}
    overhead_s = sla.queue_overhead_s + sla.fixed_overhead_s
    for query_uid, (_, query) in simulator.query_manager.finished_queries.items():
        request_id = query_uid_to_request_id[query_uid]
        metrics[request_id] = extract_helix_query_metrics(
            request_id=request_id, query=query, overhead_s=overhead_s
        )

    violation_candidates: list[tuple[float, str, str, float, float]] = []
    for request in requests:
        metric = metrics[request.id]
        if metric.aligned_ttft_s > sla.ttft_s:
            violation_candidates.append(
                (
                    request.arrival_time_s + metric.aligned_ttft_s,
                    "ttft",
                    request.id,
                    metric.aligned_ttft_s,
                    sla.ttft_s,
                )
            )
        for token_index, observed in enumerate(metric.decode_tpot_s):
            if observed > sla.tpot_s:
                # Exact simulated token finish time is not retained in the compact
                # metric, so ordering uses request arrival + cumulative token times.
                violation_candidates.append(
                    (
                        request.arrival_time_s
                        + metric.aligned_ttft_s
                        + sum(metric.decode_tpot_s[: token_index + 1]),
                        "tpot",
                        request.id,
                        observed,
                        sla.tpot_s,
                    )
                )
    violation_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    first = violation_candidates[0] if violation_candidates else None
    return HelixReferenceResult(
        feasible=first is None,
        finished_requests=len(metrics),
        total_requests=len(requests),
        final_time_s=simulator.current_time - base_time,
        first_violation_kind=first[1] if first else None,
        first_violation_request_id=first[2] if first else None,
        first_violation_observed_s=first[3] if first else None,
        first_violation_limit_s=first[4] if first else None,
        query_metrics=metrics,
    )
