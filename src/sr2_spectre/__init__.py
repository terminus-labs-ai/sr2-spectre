"""sr2-spectre — SR2 agent runtime.

Spectre owns agent identity, the tool execution loop, plugins, and sessions.
All LLM calls flow through sr2-relay — spectre never imports sr2 core directly.
"""

__version__ = "0.1.0"
