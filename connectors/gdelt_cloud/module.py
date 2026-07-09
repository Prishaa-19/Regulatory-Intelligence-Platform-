"""GDELT Cloud MCP connector.

Mounts the GDELT Cloud MCP server (https://gdelt-cloud-mcp.fastmcp.app/mcp)
over Streamable HTTP so its tools (gdelt_cloud_tool_list, gdelt_cloud_tool_get,
gdelt_cloud_tool_call, etc.) become available under the "gdelt_cloud" prefix.

Auth is a Bearer token read from GDELT_CLOUD_API_KEY, passed through
ConnectorSpec.credentials for proxy.py's _build_backend to apply as the
outbound Authorization header.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("netigen_mcp.connectors.gdelt_cloud")

if not os.getenv("GDELT_CLOUD_MCP_DISABLED"):
    from core.registry import ConnectorSpec, MCPModuleSpec, register

    _API_KEY = os.getenv("GDELT_CLOUD_API_KEY", "")

    register(MCPModuleSpec(
        name="gdelt_cloud",
        kind="connector",
        tool_specs=[],
        connector_spec=ConnectorSpec(
            id="gdelt_cloud",
            name="gdelt_cloud",
            mcp_url="https://gdelt-cloud-mcp.fastmcp.app/mcp",
            transport="http",
            credentials={"authorization_token": _API_KEY},
            enabled=bool(_API_KEY),
            workspace_id="global",
        ),
        on_startup=None,
    ))
    log.info("[gdelt_cloud] connector registered")
