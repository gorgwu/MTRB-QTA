"""Shared dataset, config, scoring, and reporting helpers for MTRB-QTA."""

from __future__ import annotations

import json
import logging
import math
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MTRB_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = MTRB_ROOT.parent
JIUWENSWARM_ROOT = WORKSPACE_ROOT / "jiuwenswarm"
AGENT_CORE_ROOT = WORKSPACE_ROOT / "agent-core"
TOOLS_PATH = MTRB_ROOT / "metatool" / "ts_metatool_api_list_rewrite.json"
QUERIES_PATH = MTRB_ROOT / "metatool" / "mtr_metatool_test.json"
RESTBENCH_TOOLS_PATH = MTRB_ROOT / "restbench" / "tmdb_tool_rewrite.json"
RESTBENCH_QUERIES_PATH = MTRB_ROOT / "restbench" / "mtr_restbench_test.json"
logger = logging.getLogger(__name__)

for _path in (AGENT_CORE_ROOT, JIUWENSWARM_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@dataclass(frozen=True)
class ToolDocument:
    mtrb_id: int
    name: str
    description: str


@dataclass(frozen=True)
class BenchmarkCase:
    index: int
    query: str
    relevant_ids: tuple[int, ...]
    relevant_names: tuple[str, ...]


def load_dataset(dataset: str = "metatool") -> tuple[list[ToolDocument], list[BenchmarkCase]]:
    """Load and validate a released MTRB catalog and its test split."""
    dataset = dataset.strip().lower()
    if dataset == "metatool":
        tools_path, queries_path = TOOLS_PATH, QUERIES_PATH
        expected_tool_count = 199
    elif dataset == "restbench":
        tools_path, queries_path = RESTBENCH_TOOLS_PATH, RESTBENCH_QUERIES_PATH
        expected_tool_count = 54
    else:
        raise ValueError(f"Unsupported MTRB dataset: {dataset!r}")
    try:
        raw_tools = json.loads(tools_path.read_text(encoding="utf-8"))
        raw_cases = json.loads(queries_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load MTRB-{dataset} data: {exc}") from exc

    if not isinstance(raw_tools, (dict, list)):
        raise ValueError(f"Tool catalog must be a JSON object or array: {tools_path}")
    if not isinstance(raw_cases, list):
        raise ValueError(f"Test split must be a JSON array: {queries_path}")

    tools: list[ToolDocument] = []
    if dataset == "metatool":
        for index, (raw_name, raw_description) in enumerate(raw_tools.items()):
            name = str(raw_name).strip()
            description = str(raw_description).strip()
            if not name or not description:
                raise ValueError(f"Tool document {index} has an empty name or description")
            tools.append(ToolDocument(index, name, description))
    else:
        for index, raw_tool in enumerate(raw_tools):
            if not isinstance(raw_tool, dict):
                raise ValueError(f"Tool document {index} must be a JSON object")
            name = str(raw_tool.get("tool_usage") or "").strip()
            description = str(
                raw_tool.get("rewrite_description")
                or raw_tool.get("tool_description")
                or ""
            ).strip()
            try:
                tool_id = int(raw_tool["ID"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"RESTBench tool document {index} has an invalid ID") from exc
            if not name or not description:
                raise ValueError(f"RESTBench tool document {index} has an empty name or description")
            tools.append(ToolDocument(tool_id, name, description))
    by_id = {tool.mtrb_id: tool for tool in tools}
    by_name = {tool.name: tool for tool in tools}
    if len(by_id) != len(tools) or len(by_name) != len(tools):
        raise ValueError(f"MTRB-{dataset} catalog contains duplicate tool IDs or names")

    cases: list[BenchmarkCase] = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Test case {index} must be a JSON object")
        query = str(raw_case.get("query") or "").strip()
        relevant_names_raw = raw_case.get("relevant Tools") or []
        if not query or not relevant_names_raw:
            raise ValueError(f"Test case {index} is missing query or relevant labels")
        relevant_names = tuple(str(value).strip() for value in relevant_names_raw)
        if dataset == "metatool":
            relevant_ids_raw = raw_case.get("relevant IDs") or []
            try:
                relevant_ids = tuple(int(value) for value in relevant_ids_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Test case {index} has a non-integer relevant ID") from exc
            if any(value not in by_id for value in relevant_ids):
                raise ValueError(f"Test case {index} references an unknown tool ID")
            names_from_ids = tuple(by_id[value].name for value in relevant_ids)
            if set(names_from_ids) != set(relevant_names):
                raise ValueError(
                    f"Test case {index} label mismatch: IDs map to {names_from_ids}, "
                    f"but relevant Tools lists {relevant_names}"
                )
        else:
            # RESTBench's released relevant IDs refer to the source API list,
            # while tmdb_tool_rewrite.json has its own reordered IDs. Normalize
            # report IDs against the actual catalog using endpoint names.
            missing_names = [name for name in relevant_names if name not in by_name]
            if missing_names:
                raise ValueError(f"RESTBench test case {index} has unknown tools: {missing_names}")
            relevant_ids = tuple(by_name[name].mtrb_id for name in relevant_names)
        cases.append(BenchmarkCase(index, query, relevant_ids, relevant_names))

    if len(tools) != expected_tool_count or len(cases) != 90:
        raise ValueError(
            f"Unexpected MTRB-{dataset} data size: {len(tools)} tools, "
            f"{len(cases)} test cases (expected {expected_tool_count} and 90)"
        )
    return tools, cases


def load_jiuwenswarm_config() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the active JiuwenSwarm config and its discovery settings."""
    from jiuwenswarm.common.config import get_config, get_tool_discovery_config
    from jiuwenswarm.common.utils import get_env_file
    from jiuwenswarm.dotenv_early import load_dotenv_runtime

    load_dotenv_runtime(dotenv_path=get_env_file(), override=True)
    config = get_config()
    if not isinstance(config, dict):
        raise RuntimeError("JiuwenSwarm config.yaml did not load as a mapping")
    return config, get_tool_discovery_config(config)


def require_backend(discovery_config: dict[str, Any], expected: str) -> None:
    actual = str(discovery_config.get("tool_discovery_backend") or "bm25").lower()
    if actual != expected:
        raise RuntimeError(
            f"JiuwenSwarm config selects tool_discovery_backend={actual!r}, "
            f"but this is the {expected.upper()} runner. Change that setting in "
            "JiuwenSwarm's config.yaml and rerun."
        )


def require_progressive_tools(config: dict[str, Any]) -> None:
    from jiuwenswarm.common.config import get_progressive_tool_enabled

    if not get_progressive_tool_enabled(config):
        raise RuntimeError(
            "JiuwenSwarm progressive_tool_enabled is false. Enable progressive "
            "tool discovery in config.yaml before running this benchmark."
        )


def require_benchmark_tool_inventory(agent: Any, tool_documents: list[ToolDocument]) -> None:
    """Fail closed if unrelated built-in tools were registered on a runner."""
    from openjiuwen.core.foundation.tool import ToolCard

    ability_manager = getattr(agent, "ability_manager", None)
    list_abilities = getattr(ability_manager, "list", None)
    if not callable(list_abilities):
        raise RuntimeError("Cannot inspect the benchmark agent's registered tools")

    cards = list_abilities() or []
    expected_catalog = {document.name for document in tool_documents}
    expected_names = expected_catalog | {"tool_search", "tool_call"}
    registered = {
        str(getattr(card, "name", "")): card
        for card in cards
        if isinstance(card, ToolCard) and getattr(card, "name", None)
    }
    unexpected = sorted(set(registered) - expected_names)
    missing_catalog = sorted(expected_catalog - set(registered))
    missing_wrappers = sorted({"tool_search", "tool_call"} - set(registered))
    wrong_exposure = []
    for name in expected_catalog & set(registered):
        exposure = getattr(registered[name], "exposure", None)
        exposure = getattr(exposure, "value", exposure)
        if str(exposure).lower() != "deferred":
            wrong_exposure.append(name)
    for name in {"tool_search", "tool_call"} & set(registered):
        exposure = getattr(registered[name], "exposure", None)
        exposure = getattr(exposure, "value", exposure)
        if str(exposure).lower() != "direct":
            wrong_exposure.append(name)

    if unexpected or missing_catalog or missing_wrappers or wrong_exposure:
        problems = []
        if unexpected:
            problems.append(f"unexpected tools registered: {unexpected}")
        if missing_catalog:
            problems.append(f"benchmark tools missing from registry: {missing_catalog}")
        if missing_wrappers:
            problems.append(f"required progressive wrappers missing: {missing_wrappers}")
        if wrong_exposure:
            problems.append(f"tools have incorrect exposure: {sorted(wrong_exposure)}")
        raise RuntimeError("Benchmark tool inventory check failed: " + "; ".join(problems))


class RuntimeSelectionCapture:
    """Instrument JiuwenSwarm's initialized ProgressiveToolRail for one query."""

    def __init__(self, rail: Any, backend: str, agent: Any):
        self.rail = rail
        self.backend = backend
        self.records: list[dict[str, Any]] = []
        if backend == "bm25":
            from openjiuwen.core.runner import Runner

            search_card = agent.ability_manager.get("tool_search")
            search_tool = (
                Runner.resource_mgr.get_tool(search_card.id)
                if search_card is not None
                else None
            )
            if search_tool is None or not callable(
                getattr(search_tool, "_search_tools", None)
            ):
                raise RuntimeError(
                    "Cannot instrument the registered tool_search executor"
                )
            original = search_tool._search_tools

            async def capture_search(query, limit=5, session=None):
                results = await original(query, limit, session)
                self.records.append({
                    "query": query,
                    "results": [
                        str(item.get("name") or "")
                        for item in results
                        if isinstance(item, dict)
                    ],
                })
                return results

            # ToolSearchTool stores the bound callback during ProgressiveToolRail.init;
            # patch the registered executor callback, not the rail method afterward.
            search_tool._search_tools = capture_search
        elif backend == "jev":
            original = rail._route_deferred_tools

            async def capture_route(ctx):
                selected, fallback = await original(ctx)
                discovery_state = (
                    ctx.session.get_state("__progressive_tool_discovery__")
                    if ctx.session is not None
                    else None
                )
                scores = (
                    discovery_state.get("scores", {})
                    if isinstance(discovery_state, dict)
                    else {}
                )
                self.records.append({
                    "query": rail._latest_user_request(ctx.inputs),
                    "results": [
                        {"name": str(item.name), "score": scores.get(str(item.name))}
                        for item in selected
                    ],
                    "error": (
                        str(discovery_state.get("fallback_error") or "Jev routing fell back to BM25")
                        if fallback and isinstance(discovery_state, dict)
                        else "Jev routing fell back to BM25" if fallback else None
                    ),
                })
                return selected, fallback

            rail._route_deferred_tools = capture_route
        else:
            raise ValueError(f"Unsupported runtime discovery backend: {backend!r}")

def replace_runtime_tool_inventory(
    agent: Any,
    tool_documents: list[ToolDocument],
    benchmark_tools: list[Any],
) -> Any:
    """Keep the default session rails while limiting tools to MTRB and wrappers."""
    from openjiuwen.core.foundation.tool import ToolCard
    from openjiuwen.harness.rails.progressive_tool_rail import ProgressiveToolRail

    ability_manager = getattr(agent, "ability_manager", None)
    if ability_manager is None:
        raise RuntimeError("Default JiuwenSwarm agent has no ability manager")
    allowed = {document.name for document in tool_documents} | {"tool_search", "tool_call"}
    extras = [
        card.name
        for card in (ability_manager.list() or [])
        if isinstance(card, ToolCard) and card.name not in allowed
    ]
    if extras:
        ability_manager.remove_ability(extras)
    for tool in benchmark_tools:
        ability_manager.add_ability(tool.card, tool)

    progressive_rails = [
        rail
        for rail in getattr(agent, "_registered_rails", [])
        if isinstance(rail, ProgressiveToolRail)
    ]
    if len(progressive_rails) != 1:
        raise RuntimeError(
            "Expected exactly one initialized ProgressiveToolRail in the default "
            f"JiuwenSwarm conversation, found {len(progressive_rails)}"
        )
    rail = progressive_rails[0]
    rail._rebuild_tool_search_index(agent)
    require_benchmark_tool_inventory(agent, tool_documents)
    return rail


class JiuwenSwarmNewConversationRuntime:
    """Create benchmark turns through JiuwenSwarm's normal new-session setup."""

    def __init__(self):
        self.swarm = None
        self.root_adapter = None

    async def initialize(self) -> None:
        from jiuwenswarm.server.runtime.agent_adapter.interface import JiuWenSwarm

        self.swarm = JiuWenSwarm()
        self.root_adapter = self.swarm._ensure_adapter(mode="agent")
        # The browser frontend uses session-scoped adapters and doesn't preload
        # global-default MCP servers into a new web conversation.
        self.root_adapter._channel_id = "web"

    async def prepare_query(
        self,
        *,
        session_id: str,
        query: str,
        tool_documents: list[ToolDocument],
        benchmark_tools: list[Any],
        backend: str,
        dataset: str,
    ) -> tuple[Any, Any, None, RuntimeSelectionCapture]:
        from jiuwenswarm.common.schema.agent import AgentRequest
        from jiuwenswarm.common.schema.message import ReqMethod

        if self.swarm is None or self.root_adapter is None:
            raise RuntimeError("Initialize the JiuwenSwarm runtime before preparing a query")
        await self.swarm.prepare_session(
            session_id=session_id,
            channel_id="web",
            mode="agent",
        )
        session_adapter = self.root_adapter._get_cached_session_adapter(session_id)
        if session_adapter is None or session_adapter._instance is None:
            raise RuntimeError("JiuwenSwarm did not create the new conversation agent")

        rail = replace_runtime_tool_inventory(
            session_adapter._instance,
            tool_documents,
            benchmark_tools,
        )
        if backend == "jev":
            # This is an experiment-only override on the fresh benchmark agent;
            # standard JiuwenSwarm sessions keep Jev's no-tool option enabled.
            rail._config.tool_discovery_include_no_tool = False
        require_benchmark_tool_inventory(session_adapter._instance, tool_documents)
        session_adapter._benchmark_tool_documents = tool_documents
        # The normal JiuwenSwarm request path refreshes session scoped tools
        # (cron, messaging, send-file) immediately before invoking the agent.
        # Re-apply this benchmark's allowlist after that refresh, so the BM25
        # catalog cannot accidentally include built-in tools outside MTRB.
        original_update_session_tools = session_adapter._update_session_tools

        async def update_session_tools_with_benchmark_inventory(*args, **kwargs):
            result = await original_update_session_tools(*args, **kwargs)
            replace_runtime_tool_inventory(
                session_adapter._instance,
                tool_documents,
                benchmark_tools,
            )
            return result

        session_adapter._update_session_tools = update_session_tools_with_benchmark_inventory
        from openjiuwen.core.single_agent.prompts.builder import PromptSection

        prompt_builder = getattr(session_adapter._instance, "system_prompt_builder", None)
        if prompt_builder is None:
            raise RuntimeError("Default JiuwenSwarm agent has no system prompt builder")
        if backend == "bm25":
            instruction_en = (
                f"{dataset} tool retrieval benchmark: call tool_search exactly once "
                "with limit=10 for the user's request. Do not call tool_call or "
                "execute a benchmark tool. The search result is the benchmark output."
            )
            instruction_cn = (
                f"{dataset} 工具检索基准：针对用户请求恰好调用一次 tool_search，"
                "limit=10。不要调用 tool_call，也不要执行基准工具。检索结果就是基准输出。"
            )
        else:
            instruction_en = (
                f"{dataset} tool retrieval benchmark: use the automatically routed "
                "deferred tools to identify the best tools for the user's request. "
                "Do not call tool_search or tool_call, and do not execute a benchmark tool."
            )
            instruction_cn = (
                f"{dataset} 工具检索基准：使用自动路由的延迟工具识别最适合用户请求的工具。"
                "不要调用 tool_search 或 tool_call，也不要执行基准工具。"
            )
        prompt_builder.add_section(
            PromptSection(
                name="mtrb_benchmark_instructions",
                content={"en": instruction_en, "cn": instruction_cn},
                priority=900,
            )
        )
        capture = RuntimeSelectionCapture(rail, backend, session_adapter._instance)
        request_id = uuid.uuid4().hex
        request = AgentRequest(
            request_id=request_id,
            channel_id="web",
            session_id=session_id,
            req_method=ReqMethod.CHAT_SEND,
            params={
                "query": query,
                "mode": "agent",
                "supports_user_interaction": False,
            },
        )
        # Let process_message build the same request inputs as the live facade
        # path after a new conversation has been prepared.
        return session_adapter, request, None, capture

    async def invoke(
        self,
        session_adapter: Any,
        request: Any,
        inputs: None,
        capture: RuntimeSelectionCapture,
    ) -> Any:
        require_benchmark_tool_inventory(
            session_adapter._instance,
            getattr(session_adapter, "_benchmark_tool_documents", []),
        )
        capture.records.clear()
        if self.swarm is None:
            raise RuntimeError("JiuwenSwarm runtime is not initialized")
        return await self.swarm.process_message(request)

    async def cleanup_query(self, session_id: str) -> None:
        """Release a fresh conversation after its single benchmark turn."""
        if self.root_adapter is None:
            return
        cleanup = getattr(self.root_adapter, "cleanup_session_adapter", None)
        if callable(cleanup):
            try:
                await cleanup(session_id)
            except Exception:
                logger.exception("Failed to clean benchmark conversation %s", session_id)

    async def cleanup(self) -> None:
        if self.swarm is not None:
            await self.swarm.cleanup()
            self.swarm = None
            self.root_adapter = None


def agent_response_text(response: Any) -> str | None:
    """Return the final text field from the normal JiuwenSwarm response."""
    payload = getattr(response, "payload", None)
    if not isinstance(payload, dict):
        return None
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "output", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value
    return None


def configured_agent_model(config: dict[str, Any]) -> str:
    models = config.get("models") if isinstance(config.get("models"), dict) else {}
    default = models.get("default") if isinstance(models.get("default"), dict) else {}
    client = (
        default.get("model_client_config")
        if isinstance(default.get("model_client_config"), dict)
        else {}
    )
    return str(default.get("model_name") or client.get("model_name") or "JiuwenSwarm default model")


def selected_cases(
    cases: list[BenchmarkCase], num_queries: int | None
) -> list[BenchmarkCase]:
    return cases if num_queries is None else cases[:num_queries]


def compute_metrics(
    rows: list[dict[str, Any]], ks: tuple[int, ...] = (5, 10)
) -> dict[str, float]:
    """Compute MTRB Sufficiency@k and binary NDCG@k over all requested rows."""
    if not rows:
        return {f"{metric}@{k}": 0.0 for k in ks for metric in ("S", "NDCG")}
    metrics: dict[str, float] = {}
    for k in ks:
        sufficiency = 0.0
        ndcg = 0.0
        for row in rows:
            relevant = set(row["relevant_tools"])
            ranked = list(row.get("selected_tools") or [])[:k]
            relevant_ranks = [rank for rank, name in enumerate(ranked, 1) if name in relevant]
            # MTRB Sufficiency@k is all-required coverage: for tasks whose
            # annotation lists multiple tools, every one must occur in top-k.
            if relevant and relevant.issubset(set(ranked)):
                sufficiency += 1.0
            if relevant_ranks:
                dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
                ideal_count = min(k, len(relevant))
                idcg = sum(
                    1.0 / math.log2(rank + 1)
                    for rank in range(1, ideal_count + 1)
                )
                ndcg += dcg / idcg if idcg else 0.0
        metrics[f"S@{k}"] = sufficiency / len(rows)
        metrics[f"NDCG@{k}"] = ndcg / len(rows)
    return metrics


def write_report(
    *,
    output: str | Path,
    title: str,
    backend: str,
    description: str,
    model: str,
    config_summary: dict[str, Any],
    rows: list[dict[str, Any]],
    total_dataset_size: int,
    started_at: datetime,
    dataset_name: str = "metatool",
) -> Path:
    """Write a readable report with JSON records for machine parsing."""
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = MTRB_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(rows)
    errors = sum(bool(row.get("error")) for row in rows)
    id_to_name = {
        tool.mtrb_id: tool.name for tool in load_dataset(dataset_name)[0]
    }
    dataset_title = {
        "metatool": "MetaTool",
        "restbench": "RestBench",
    }.get(dataset_name.lower(), dataset_name.title())

    lines = [
        title,
        f"Run status: completed ({errors} query errors)",
        f"Started (local): {started_at.astimezone().isoformat(timespec='seconds')}",
        f"Benchmark: MTRB-{dataset_title}; test queries: {len(rows)}/{total_dataset_size}",
        f"Backend: {backend}",
        f"Route: {description}",
        f"Configured model: {model}",
        "Configuration: " + json.dumps(config_summary, ensure_ascii=False, sort_keys=True),
        f"Queries completed without infrastructure errors: {len(rows) - errors}/{len(rows)}",
        f"Query errors: {errors}",
        "Metrics use every requested query as the denominator; failed queries count as misses.",
        f"Sufficiency@5: {metrics['S@5']:.4f} ({metrics['S@5']:.1%})",
        f"Sufficiency@10: {metrics['S@10']:.4f} ({metrics['S@10']:.1%})",
        f"NDCG@5: {metrics['NDCG@5']:.4f}",
        f"NDCG@10: {metrics['NDCG@10']:.4f}",
        "Per-query agent_output_if_no_tool_call stores the final assistant text "
        "when an answer completes without a tool call; null means no such answer was returned.",
        "",
        "Per-query results (one JSON object per line):",
    ]

    for row in rows:
        ranked = []
        for rank, raw_result in enumerate(row.get("ranked_results") or [], 1):
            if isinstance(raw_result, str):
                name, score = raw_result, None
            else:
                name = str(raw_result.get("name") or "")
                score = raw_result.get("score")
            mtrb_id = next(
                (tool_id for tool_id, tool_name in id_to_name.items() if tool_name == name),
                None,
            )
            ranked.append({"rank": rank, "id": mtrb_id, "name": name, "score": score})
        relevant = set(row["relevant_tools"])
        hit_ranks = [item["rank"] for item in ranked if item["name"] in relevant]
        serialized = {
            "index": row["index"],
            "query": row["query"],
            "router_query": row.get("router_query"),
            "relevant_ids": row["relevant_ids"],
            "relevant_tools": row["relevant_tools"],
            "ranked_results": ranked,
            "relevant_ranks": hit_ranks,
            "additional_searches": row.get("additional_searches", []),
            "agent_output_if_no_tool_call": row.get(
                "agent_output_if_no_tool_call"
            ),
            "error": row.get("error"),
        }
        lines.append(json.dumps(serialized, ensure_ascii=False, separators=(",", ":")))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def row_for_case(case: BenchmarkCase) -> dict[str, Any]:
    return {
        "index": case.index,
        "query": case.query,
        "relevant_ids": list(case.relevant_ids),
        "relevant_tools": list(case.relevant_names),
        "router_query": None,
        "ranked_results": [],
        "selected_tools": [],
        "agent_output_if_no_tool_call": None,
        "error": None,
    }
