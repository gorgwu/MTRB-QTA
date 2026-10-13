"""Run MTRB-MetaTool through JiuwenSwarm's System One rail using local Kev."""

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
    selected_tool_catalog,
    write_report,
)


DEFAULT_OUTPUT = "mtrb_metatool_kev_results.txt"


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
        "--query-index",
        type=int,
        help="run one zero-based query index from the test split (for example, 1 for the Leo query)",
    )
    parser.add_argument(
        "--num-tools",
        type=int,
        help="limit the candidate catalog for smoke runs; selected queries' gold tools are always retained",
    )
    parser.add_argument("--catalog-seed", type=int, default=42, help="seed for deterministic smoke distractors")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="report path")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=300.0,
        help="System One HTTP timeout in seconds (default: 300; CPU Kev may need longer)",
    )
    return parser.parse_args()


def _install_request_timeout(timeout_seconds: float):
    """Apply a runner-local timeout to calls made by ProgressiveToolRail."""
    from openjiuwen.harness.rails import progressive_tool_rail

    original = progressive_tool_rail.rank_decisions_api_tools

    async def rank_with_timeout(**kwargs):
        kwargs["request_timeout"] = timeout_seconds
        return await original(**kwargs)

    progressive_tool_rail.rank_decisions_api_tools = rank_with_timeout
    return original


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

        tools.append(LocalFunction(card=card, func=retrieval_only_executor))
    return tools


async def _run(args: argparse.Namespace) -> Path:
    from jiuwenswarm.symphony.llm import LLMConfig

    started_at = datetime.now().astimezone()
    if args.request_timeout <= 0:
        raise ValueError("request-timeout must be greater than zero")
    config, discovery = load_jiuwenswarm_config()
    require_backend(discovery, "local")
    require_progressive_tools(config)

    kev_model = str(discovery.get("tool_discovery_model") or "")
    api_base = str(discovery.get("tool_discovery_api_base") or "")
    if "kev" not in kev_model.lower():
        raise RuntimeError(
            "Kev runner requires MTRB_TOOL_DISCOVERY_MODEL=kev-latest "
            "(or another Kev model name)."
        )
    if not api_base:
        raise RuntimeError(
            "Kev runner requires MTRB_TOOL_DISCOVERY_API_BASE pointing to the "
            "Kev /v1/systemone endpoint."
        )

    api_key = discovery.get("tool_discovery_api_key") or os.getenv(
        "TOOL_DISCOVERY_API_KEY"
    )
    if not str(api_key or "").strip():
        raise RuntimeError(
            "Kev requires MTRB_TOOL_DISCOVERY_API_KEY or a configured "
            "tool discovery API key."
        )

    tool_documents, all_cases = load_dataset()
    if args.query_index is not None:
        if args.query_index < 0 or args.query_index >= len(all_cases):
            raise ValueError(
                f"query-index must be between 0 and {len(all_cases) - 1}"
            )
        cases = [all_cases[args.query_index]]
    else:
        cases = selected_cases(all_cases, args.num_queries)
    original_tool_count = len(tool_documents)
    tool_documents = selected_tool_catalog(
        tool_documents, cases, args.num_tools, seed=args.catalog_seed
    )
    llm_config = LLMConfig.from_default_model()
    router_name = "Kev"
    max_tools = int(discovery.get("tool_discovery_max_tools") or 10)
    min_score = float(discovery.get("tool_discovery_min_score", 0.0))
    api_base = discovery.get("tool_discovery_api_base")
    runtime = JiuwenSwarmNewConversationRuntime()
    await runtime.initialize()
    original_rank = _install_request_timeout(args.request_timeout)
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
                    dataset="MTRB-MetaTool",
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
                    row["error"] = "Kev-compatible System One rail did not complete tool selection."
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
        from openjiuwen.harness.rails import progressive_tool_rail

        progressive_tool_rail.rank_decisions_api_tools = original_rank

    return write_report(
        output=args.output,
        title=f"MTRB-MetaTool {router_name} retrieval benchmark",
        backend=f"JiuwenSwarm ProgressiveToolRail {router_name} auto-discovery",
        description=(
            f"Runs automatic {router_name} selection through the Jev-compatible "
            "System One adapter in the JiuwenSwarm DeepAgent rail. "
            "When the agent completes without a tool call, its text output is "
            "included in that query record."
        ),
        model=f"agent: {llm_config.model}; {router_name} router: {kev_model}",
        config_summary={
            "tool_discovery_backend": "local",
            "tool_discovery_model": kev_model,
            "tool_discovery_max_tools": max_tools,
            "tool_discovery_min_score": min_score,
            "decision_request_timeout_sec": args.request_timeout,
            "tool_discovery_include_no_tool": False,
            "tool_discovery_api_base": api_base or "default Decisions API",
            "tool_count": len(tool_documents),
            "full_catalog_tool_count": original_tool_count,
            "catalog_seed": args.catalog_seed if args.num_tools is not None else None,
            "catalog_scope": "reduced smoke catalog; all selected gold tools retained"
            if len(tool_documents) < original_tool_count
            else "full dataset catalog",
            "tool_inventory_policy": "MTRB catalog deferred; only tool_search and tool_call direct",
            "answer_generation_after_selection": True,
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
        raise SystemExit(f"Kev benchmark setup failed: {type(exc).__name__}: {exc}") from exc
    print(f"Report written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
