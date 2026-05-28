"""MCPClient — connects to an MCP server (stdio or http) and returns MCPToolBridges."""
from __future__ import annotations

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client

from sr2_spectre.mcp.tool_bridge import MCPToolBridge


class MCPConnectionError(Exception):
    """Raised when the MCP transport or session handshake fails."""


class MCPClient:
    """Connects to an MCP server and exposes its tools as MCPToolBridge instances.

    Args:
        server_type: "stdio" or "http"
        **transport_kwargs:
            stdio: command (list[str]), args (list[str]=[]), env (dict={})
            http:  url (str)
    """

    def __init__(self, server_type: str, **transport_kwargs: object) -> None:
        self._server_type = server_type
        self._transport_kwargs = transport_kwargs
        # Held open between connect() and close()
        self._transport_ctx = None
        self._session_ctx = None

    async def connect(self) -> list[MCPToolBridge]:
        """Open the transport, initialise the session, and return one bridge per tool."""
        try:
            if self._server_type == "stdio":
                command = self._transport_kwargs.get("command", "")
                args = self._transport_kwargs.get("args", [])
                env = self._transport_kwargs.get("env", {})
                # command may be a str (real usage) or a list (test stubs).
                # StdioServerParameters requires a str; when mocked the
                # validation never runs, so we normalise here.
                if isinstance(command, list):
                    cmd_str = command[0] if command else ""
                    extra_args = list(command[1:]) + list(args)
                else:
                    cmd_str = command
                    extra_args = list(args)
                params = StdioServerParameters(command=cmd_str, args=extra_args, env=env or None)
                transport_ctx = stdio_client(params)
            else:
                url = self._transport_kwargs["url"]
                transport_ctx = sse_client(url=url)

            self._transport_ctx = transport_ctx
            read, write = await transport_ctx.__aenter__()

            session_ctx = ClientSession(read, write)
            self._session_ctx = session_ctx
            session = await session_ctx.__aenter__()

            await session.initialize()
            result = await session.list_tools()

        except MCPConnectionError:
            raise
        except Exception as exc:
            raise MCPConnectionError(str(exc)) from exc

        return [
            MCPToolBridge(mcp_tool=tool, call_tool=session.call_tool)
            for tool in result.tools
        ]

    async def close(self) -> None:
        """Exit session and transport context managers. Safe to call before connect()."""
        if self._session_ctx is not None:
            await self._session_ctx.__aexit__(None, None, None)
            self._session_ctx = None

        if self._transport_ctx is not None:
            await self._transport_ctx.__aexit__(None, None, None)
            self._transport_ctx = None
