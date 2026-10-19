"""HybridAgent package (lazy imports)."""

from __future__ import annotations

_LAZY_ATTRS = {
    "ANCHOR_TOKENS": "hybrid_agent.constants",
    "ENTRY_ANCHOR": "hybrid_agent.constants",
    "EXIT_ANCHOR": "hybrid_agent.constants",
    "special_token_id": "hybrid_agent.constants",
    "extract_answer": "hybrid_agent.answer",
    "strip_latent_spans": "hybrid_agent.answer",
    "HybridAgentModel": "hybrid_agent.model",
    "add_anchor_tokens": "hybrid_agent.model",
    "load_hybrid_agent": "hybrid_agent.model",
    "resolve_model_dir": "hybrid_agent.model",
    "HybridGenerationEngine": "hybrid_agent.engine",
    "RowState": "hybrid_agent.engine",
    "DECODE": "hybrid_agent.engine",
    "LATENT": "hybrid_agent.engine",
    "FEED": "hybrid_agent.engine",
    "IDLE": "hybrid_agent.engine",
    "DONE": "hybrid_agent.engine",
    "HybridLink": "hybrid_agent.hybridlink",
    "LatentSpanReadout": "hybrid_agent.readout",
    "latent_readout": "hybrid_agent.readout",
    "render_with_readout": "hybrid_agent.readout",
    "BaseTool": "hybrid_agent.tools",
    "PythonTool": "hybrid_agent.tools",
    "RetryingSearchTool": "hybrid_agent.tools",
    "TavilySearchTool": "hybrid_agent.tools",
    "ToolExecutor": "hybrid_agent.tools",
    "build_tool_executor": "hybrid_agent.tools",
    "SYSTEM_PROMPT": "hybrid_agent.prompts",
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name):
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(name)
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals()) + list(_LAZY_ATTRS))
