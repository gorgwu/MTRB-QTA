"""Run MTRB-MetaTool through JiuwenSwarm's configured BM25 tool_search path."""

from __future__ import annotations

import argparse
import asyncio
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


DEFAULT_OUTPUT = "mtrb_metatool_bm25_results.txt"


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
            id=f"mtrb_metatool_{document.mtrb_id}",
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
                error="MTRB-MetaTool is retrieval-only; tool execution is disabled.",
            )

        # BM25ToolIndex consumes ToolInfo parameters, and the production
        # catalog is formed from each deferred card's declared schema.
        card.input_params = {"type": "object", "properties": {}}
        tools.append(LocalFunction(card=card, func=retrieval_only_executor))
    return tools


async def _run(args: argparse.Namespace) -> Path:
    from jiuwenswarm.symphony.llm import LLMConfig

    started_at = datetime.now().astimezone()
    config, discovery = load_jiuwenswarm_config()
    require_backend(discovery, "bm25")
    require_progressive_tools(config)

    tool_documents, all_cases = load_dataset()
    cases = selected_cases(all_cases, args.num_queries)
    llm_config = LLMConfig.from_default_model()
    runtime = JiuwenSwarmNewConversationRuntime()
    await runtime.initialize()
    benchmark_tools = _make_benchmark_tools(tool_documents)
    rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            row = row_for_case(case)
            session_id = uuid.uuid4().hex
            try:
                session_adapter, request, inputs, capture = await runtime.prepare_query(
                    session_id=session_id,
                    query=case.query,
                    tool_documents=tool_documents,
                    benchmark_tools=benchmark_tools,
                    backend="bm25",
                    dataset="MTRB-MetaTool",
                )
                agent_result = await runtime.invoke(session_adapter, request, inputs, capture)
                if capture.records:
                    first_search = capture.records[0]
                    row["router_query"] = first_search["query"]
                    row["ranked_results"] = first_search["results"]
                    row["selected_tools"] = first_search["results"]
                    row["additional_searches"] = capture.records[1:]
                else:
                    row["agent_output_if_no_tool_call"] = agent_response_text(agent_result)
                    row["error"] = "Agent did not call tool_search during its first turn."
            except Exception as exc:  # keep the rest of the benchmark running
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
        title="MTRB-MetaTool BM25 retrieval benchmark",
        backend="JiuwenSwarm BM25 via ProgressiveToolRail.tool_search",
        description=(
            "JiuWenSwarm-configured agent model chooses the tool_search query; "
            "the first actual BM25 result list is scored. Tool execution is disabled."
        ),
        model=llm_config.model,
        config_summary={
            "tool_discovery_backend": "bm25",
            "tool_search_limit": 10,
            "tool_count": len(tool_documents),
            "tool_inventory_policy": "MTRB catalog deferred; only tool_search and tool_call direct",
            "capture_text_when_no_tool_call": True,
        },
        rows=rows,
        total_dataset_size=len(all_cases),
        started_at=started_at,
    )


def main() -> int:
    args = _arguments()
    try:
        output = asyncio.run(_run(args))
    except Exception as exc:
        raise SystemExit(f"BM25 benchmark setup failed: {type(exc).__name__}: {exc}") from exc
    print(f"Report written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
