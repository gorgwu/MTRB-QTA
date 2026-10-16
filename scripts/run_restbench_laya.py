"""Run MTRB-RestBench through a local Laya System One endpoint."""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from mtrb_common import (
    JiuwenSwarmNewConversationRuntime,
    agent_response_text,
    load_dataset,
    load_jiuwenswarm_config,
    require_backend,
    require_progressive_tools,
    row_for_case,
    selected_cases,
    write_report,
)


DEFAULT_OUTPUT = "mtrb_restbench_laya_results.txt"


def _parse_num_queries(value: str) -> int | None:
    if value.strip().lower() == "all":
        return None
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use a positive integer or 'all'") from exc
    if count < 1:
        raise argparse.ArgumentTypeError("num-queries must be at least 1")
    return count


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-queries",
        type=_parse_num_queries,
        default=None,
        help="run the first N test queries, or all 90 (default: all)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="System One HTTP timeout in seconds (default: 120)",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="report path")
    return parser.parse_args()


def _make_benchmark_tools(tool_documents: list[Any]) -> list[Any]:
    from openjiuwen.core.foundation.tool import LocalFunction, ToolCard, ToolExposure
    from openjiuwen.core.foundation.tool.schema import ToolOutput

    tools = []
    for document in tool_documents:
        card = ToolCard(
            id=f"mtrb_restbench_{document.mtrb_id}",
            name=document.name,
            description=document.description,
            input_params={"type": "object", "properties": {}},
            exposure=ToolExposure.DEFERRED,
            parallel_safe=True,
            stateless=True,
            idempotent=True,
        )

        def retrieval_only_executor(**_kwargs: Any) -> ToolOutput:
            return ToolOutput(
                success=False,
                error="MTRB-RestBench is retrieval-only; tool execution is disabled.",
            )

        tools.append(LocalFunction(card=card, func=retrieval_only_executor))
    return tools


async def _run(args: argparse.Namespace) -> Path:
    from jiuwenswarm.symphony.llm import LLMConfig

    started_at = datetime.now().astimezone()
    if args.request_timeout <= 0:
        raise ValueError("request-timeout must be greater than zero")

    os.environ.setdefault("MTRB_TOOL_DISCOVERY_MODEL", "english")
    os.environ.setdefault("MTRB_TOOL_DISCOVERY_MAX_LEN", "8192")
    os.environ.setdefault("MTRB_TOOL_DISCOVERY_HEAD_MAX_LEN", "7800")
    config, discovery = load_jiuwenswarm_config()
    require_backend(discovery, "local")
    require_progressive_tools(config)

    api_key = discovery.get("tool_discovery_api_key") or os.getenv(
        "TOOL_DISCOVERY_API_KEY"
    )
    api_base = discovery.get("tool_discovery_api_base") or os.getenv(
        "TOOL_DISCOVERY_API_BASE"
    )
    if not str(api_key or "").strip():
        raise RuntimeError(
            "Local System One routing requires TOOL_DISCOVERY_API_KEY or "
            "MTRB_TOOL_DISCOVERY_API_KEY."
        )
    if not str(api_base or "").strip():
        raise RuntimeError(
            "Local System One routing requires TOOL_DISCOVERY_API_BASE or "
            "MTRB_TOOL_DISCOVERY_API_BASE pointing to Laya."
        )

    tool_documents, all_cases = load_dataset("restbench")
    cases = selected_cases(all_cases, args.num_queries)
    llm_config = LLMConfig.from_default_model()
    router_model = str(discovery.get("tool_discovery_model") or "english")
    max_tools = int(discovery.get("tool_discovery_max_tools") or 10)
    min_score = float(discovery.get("tool_discovery_min_score", 0.0))
    runtime = JiuwenSwarmNewConversationRuntime()
    await runtime.initialize()

    from openjiuwen.harness.rails import progressive_tool_rail

    original_rank = progressive_tool_rail.rank_decisions_api_tools

    async def rank_with_laya_limits(**kwargs):
        kwargs["request_timeout"] = args.request_timeout
        for field, env_name in (
            ("max_len", "MTRB_TOOL_DISCOVERY_MAX_LEN"),
            ("head_max_len", "MTRB_TOOL_DISCOVERY_HEAD_MAX_LEN"),
        ):
            value = os.getenv(env_name)
            if value:
                kwargs[field] = int(value)
        return await original_rank(**kwargs)

    progressive_tool_rail.rank_decisions_api_tools = rank_with_laya_limits
    benchmark_tools = _make_benchmark_tools(tool_documents)
    rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            row = row_for_case(case)
            row["router_query"] = case.query
            session_id = uuid.uuid4().hex
            try:
                session_adapter, request, inputs, capture = await runtime.prepare_query(
                    session_id=session_id,
                    query=case.query,
                    tool_documents=tool_documents,
                    benchmark_tools=benchmark_tools,
                    backend="local",
                    dataset="MTRB-RestBench",
                )
                agent_result = await runtime.invoke(
                    session_adapter, request, inputs, capture
                )
                if capture.records:
                    selection = capture.records[0]
                    row["ranked_results"] = selection["results"]
                    row["selected_tools"] = [
                        item["name"] for item in selection["results"]
                    ]
                    if selection.get("error"):
                        row["error"] = selection["error"]
                    row["agent_output_if_no_tool_call"] = agent_response_text(
                        agent_result
                    )
                else:
                    row["agent_output_if_no_tool_call"] = agent_response_text(
                        agent_result
                    )
                    row["error"] = "Laya rail did not complete tool selection."
            except Exception as exc:  # keep remaining benchmark queries running
                row["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                await runtime.cleanup_query(session_id)
            rows.append(row)
            print(
                f"[{len(rows)}/{len(cases)}] query={case.index} "
                f"retrieved={len(row['selected_tools'])} error={bool(row['error'])}",
                flush=True,
            )
    finally:
        try:
            await runtime.cleanup()
        finally:
            progressive_tool_rail.rank_decisions_api_tools = original_rank

    return write_report(
        output=args.output,
        title="MTRB-RestBench Laya retrieval benchmark",
        backend="JiuwenSwarm ProgressiveToolRail Laya auto-discovery",
        description=(
            "Runs automatic Laya selection through the configured local System One "
            "endpoint. Tool execution is disabled; the report records selected "
            "tools and any final assistant text."
        ),
        model=f"agent: {llm_config.model}; Laya router: {router_model}",
        config_summary={
            "tool_discovery_backend": "local",
            "tool_discovery_model": router_model,
            "tool_discovery_max_len": os.getenv("MTRB_TOOL_DISCOVERY_MAX_LEN"),
            "tool_discovery_head_max_len": os.getenv(
                "MTRB_TOOL_DISCOVERY_HEAD_MAX_LEN"
            ),
            "tool_discovery_max_tools": max_tools,
            "tool_discovery_min_score": min_score,
            "tool_discovery_include_no_tool": False,
            "tool_discovery_api_base": api_base,
            "decision_request_timeout_sec": args.request_timeout,
            "tool_count": len(tool_documents),
            "tool_inventory_policy": "RestBench catalog deferred; only tool_search and tool_call direct",
            "answer_generation_after_selection": True,
            "capture_text_when_no_tool_call": True,
        },
        rows=rows,
        total_dataset_size=len(all_cases),
        started_at=started_at,
        dataset_name="restbench",
    )


def main() -> int:
    args = _arguments()
    try:
        output = asyncio.run(_run(args))
    except Exception as exc:
        raise SystemExit(
            f"Laya RestBench benchmark setup failed: {type(exc).__name__}: {exc}"
        ) from exc
    print(f"Report written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
