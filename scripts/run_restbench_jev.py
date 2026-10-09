"""Run MTRB-RestBench through JiuwenSwarm's configured Jev runtime rail."""

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


DEFAULT_OUTPUT = "mtrb_restbench_jev_results.txt"


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
    config, discovery = load_jiuwenswarm_config()
    require_backend(discovery, "jev")
    require_progressive_tools(config)

    api_key = discovery.get("tool_discovery_api_key") or os.getenv(
        "TOOL_DISCOVERY_API_KEY"
    )
    if not str(api_key or "").strip():
        raise RuntimeError(
            "Jev requires TOOL_DISCOVERY_API_KEY. Set it in JiuwenSwarm's "
            "config.yaml or config/.env."
        )

    tool_documents, all_cases = load_dataset("restbench")
    cases = selected_cases(all_cases, args.num_queries)
    llm_config = LLMConfig.from_default_model()
    jev_model = str(discovery.get("tool_discovery_model") or "typesafe/jev-1.13")
    max_tools = int(discovery.get("tool_discovery_max_tools") or 10)
    min_score = float(discovery.get("tool_discovery_min_score", 0.01))
    api_base = discovery.get("tool_discovery_api_base")
    runtime = JiuwenSwarmNewConversationRuntime()
    await runtime.initialize()
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
                    backend="jev",
                    dataset="MTRB-RestBench",
                )
                agent_result = await runtime.invoke(session_adapter, request, inputs, capture)
                if capture.records:
                    selection = capture.records[0]
                    row["ranked_results"] = selection["results"]
                    row["selected_tools"] = [item["name"] for item in selection["results"]]
                    if selection.get("error"):
                        row["error"] = selection["error"]
                    row["agent_output_if_no_tool_call"] = agent_response_text(agent_result)
                else:
                    row["agent_output_if_no_tool_call"] = agent_response_text(agent_result)
                    row["error"] = "Jev rail did not complete tool selection."
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
        await runtime.cleanup()

    return write_report(
        output=args.output,
        title="MTRB-RestBench Jev retrieval benchmark",
        backend="JiuwenSwarm ProgressiveToolRail Jev auto-discovery",
        description=(
            "Runs automatic Jev selection in the JiuwenSwarm DeepAgent rail. "
            "When the agent completes without a tool call, its text output is "
            "included in that query record."
        ),
        model=f"agent: {llm_config.model}; Jev router: {jev_model}",
        config_summary={
            "tool_discovery_backend": "jev",
            "tool_discovery_model": jev_model,
            "tool_discovery_max_tools": max_tools,
            "tool_discovery_min_score": min_score,
            "tool_discovery_include_no_tool": False,
            "tool_discovery_api_base": api_base or "default Decisions API",
            "tool_count": len(tool_documents),
            "tool_inventory_policy": "MTRB catalog deferred; only tool_search and tool_call direct",
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
        raise SystemExit(f"Jev benchmark setup failed: {type(exc).__name__}: {exc}") from exc
    print(f"Report written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
