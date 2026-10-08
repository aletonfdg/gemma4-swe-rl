# sim/context.py
import os
import time
import json
import numpy as np
import networkx as nx


class SwegemmaContext:
    def __init__(self, workspace, budget, harness, graph_path=None, embeddings_path=None):
        self.workspace = workspace
        self.budget = budget
        self.harness = harness
        self.tool_calls_used = 0
        self.turns_used = 0
        self.patch_submitted = False
        self.submitted_patch = ""
        self.task_start_time = time.time()
        self.agent_start_time = None
        self.graph = None
        self.node_text = {}
        self.embeddings = {}
        if graph_path and os.path.exists(graph_path):
            self._load_graph(graph_path)
        if embeddings_path and os.path.exists(embeddings_path):
            self._load_embeddings(embeddings_path)

    def start_agent_session(self):
        self.agent_start_time = time.time()

    def remaining_time_seconds(self):
        start = self.agent_start_time or self.task_start_time
        return max(0.0, self.budget.time_minutes * 60.0 - (time.time() - start))

    def agent_elapsed_seconds(self):
        start = self.agent_start_time or self.task_start_time
        return time.time() - start

    def remaining_tool_calls(self):
        return max(0, self.budget.tool_calls - self.tool_calls_used)

    def budget_exhausted(self):
        return (self.remaining_tool_calls() <= 0 or self.remaining_time_seconds() <= 0
                or self.turns_used >= self.budget.turns)

    def resolve(self, filepath):
        if filepath is None or "\x00" in filepath:
            raise ValueError("Invalid path")
        p = filepath
        if p.startswith("/workspace/"):
            p = p[len("/workspace/"):]
        elif p.startswith("/workspace"):
            p = p[len("/workspace"):].lstrip("/")
        p = p.lstrip("/")
        norm = os.path.normpath(p)
        if norm == ".." or norm.startswith(".." + os.sep):
            raise ValueError("Path traversal forbidden")
        full = os.path.normpath(os.path.join(self.workspace, norm))
        root = os.path.normpath(self.workspace)
        if not (full == root or full.startswith(root + os.sep)):
            raise ValueError("Path traversal outside /workspace forbidden")
        return full

    def _load_graph(self, path):
        data = json.load(open(path, "r", encoding="utf-8"))
        self.graph = nx.node_link_graph(data, directed=data.get("directed", True),
                                         multigraph=data.get("multigraph", True))
        for n, attrs in self.graph.nodes(data=True):
            if "text" in attrs:
                self.node_text[n] = attrs["text"]

    def _load_embeddings(self, path):
        with np.load(path, allow_pickle=True) as npz:
            for key in npz.files:
                self.embeddings[key] = npz[key].astype(np.float32)

    def resolve_node(self, name):
        if self.graph is None:
            return None
        nodes = list(self.graph.nodes)
        if name in nodes:
            return name
        for n in nodes:
            if n.endswith("." + name) or n.endswith("/" + name):
                return n
        lname = name.lower()
        for n in nodes:
            if n.lower() == lname:
                return n
        for n in nodes:
            if lname in n.lower():
                return n
        return None

    def has_code_intel(self):
        return self.graph is not None or bool(self.embeddings)
