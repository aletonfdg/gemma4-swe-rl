# sim/tool_schemas.py
ANALYZER_TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Reads a file from /workspace with 1-indexed inclusive line slicing.",
        "parameters": {"type": "object", "properties": {
            "filepath": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"}}, "required": ["filepath"]}}},
    {"type": "function", "function": {
        "name": "search_similar_code",
        "description": "Finds top-k graph nodes most similar to query via embeddings.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "k": {"type": "integer", "default": 10}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_code_neighbors",
        "description": "Finds incoming/outgoing neighbors of a symbol in the code graph.",
        "parameters": {"type": "object", "properties": {
            "node": {"type": "string"},
            "edge_type": {"type": "string"},
            "max_neighbors": {"type": "integer", "default": 50}}, "required": ["node"]}}},
    {"type": "function", "function": {
        "name": "get_code_subgraph",
        "description": "Extracts the induced subgraph for a list of symbols.",
        "parameters": {"type": "object", "properties": {
            "nodes": {"type": "array", "items": {"type": "string"}}}, "required": ["nodes"]}}},
]

# Синтетический tool, представляющий ADK agent_tool-делегирование в root-агенте.
# ADK AgentTool обычно принимает единственный текстовый запрос и прогоняет под-агента
# до завершения, возвращая его финальный текстовый ответ (с skip_summarization=true --
# БЕЗ дополнительной суммаризации этим ответом со стороны родительской модели).
CODE_ANALYZER_DELEGATION_SCHEMA = {
    "type": "function", "function": {
        "name": "code_analyzer_agent",
        "description": ("Analyzes repository source files and symbol graphs to locate root "
                         "causes. Delegate here instead of reading many files yourself."),
        "parameters": {"type": "object", "properties": {
            "request": {"type": "string",
                        "description": "What to investigate (e.g. the problem statement or "
                                        "a specific symbol/behavior to trace)."}},
            "required": ["request"]}}}

ROOT_TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Executes a shell command in /bin/bash -c inside /workspace.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Reads a file from /workspace with 1-indexed inclusive line slicing.",
        "parameters": {"type": "object", "properties": {
            "filepath": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"}}, "required": ["filepath"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replaces old_string with new_string in an existing non-empty file.",
        "parameters": {"type": "object", "properties": {
            "filepath": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "allow_multiple": {"type": "boolean", "default": False}},
            "required": ["filepath", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Creates or overwrites a file at /workspace/<filepath>.",
        "parameters": {"type": "object", "properties": {
            "filepath": {"type": "string"}, "content": {"type": "string"}},
            "required": ["filepath", "content"]}}},
    {"type": "function", "function": {
        "name": "submit_patch",
        "description": "Finalizes and submits the current working-tree diff as the solution.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_status",
        "description": "Returns live budget consumption and patch status.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_code_neighbors",
        "description": "Finds incoming/outgoing neighbors of a symbol in the code graph.",
        "parameters": {"type": "object", "properties": {
            "node": {"type": "string"},
            "edge_type": {"type": "string"},
            "max_neighbors": {"type": "integer", "default": 50}}, "required": ["node"]}}},
    {"type": "function", "function": {
        "name": "search_similar_code",
        "description": "Finds top-k graph nodes most similar to query via embeddings.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "k": {"type": "integer", "default": 10}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_code_subgraph",
        "description": "Extracts the induced subgraph for a list of symbols.",
        "parameters": {"type": "object", "properties": {
            "nodes": {"type": "array", "items": {"type": "string"}}}, "required": ["nodes"]}}},
    CODE_ANALYZER_DELEGATION_SCHEMA,
]

TOOL_NAMES_COUNTING_BUDGET = {
    "run_command", "read_file", "edit_file", "write_file",
    "get_code_neighbors", "search_similar_code", "get_code_subgraph",
}

# Обратная совместимость с уже написанным кодом (sft/train_sft.py и т.п.)
TOOL_SCHEMAS = ROOT_TOOL_SCHEMAS