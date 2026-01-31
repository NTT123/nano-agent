"""Interactive HTML viewer for Node graphs."""

import json
import sys
from pathlib import Path
from typing import Any

from nano_agent import (
    Message,
    Node,
    Role,
    StopReason,
    SubGraph,
    SystemPrompt,
    TextContent,
    ToolDefinitions,
    ToolExecution,
    ToolUseContent,
)


def to_html(node: Node, meta: dict[str, Any]) -> str:
    """Generate interactive HTML visualization."""
    ancestors = node.ancestors()

    # Build nodes and edges for D3
    nodes_data = []
    edges_data = []

    for n in ancestors:
        node_info = {
            "id": n.id,
            "type": get_type(n),
            "label": get_label(n),
            "timestamp": n.timestamp,
            "metadata": n.metadata,
            "data": serialize_data(n.data),
            "parent_count": len(n.parents),
        }
        nodes_data.append(node_info)

        for parent in n.parents:
            edges_data.append({"source": parent.id, "target": n.id})

    graph_data = {
        "meta": meta,
        "nodes": nodes_data,
        "edges": edges_data,
    }

    return HTML_TEMPLATE.replace(
        "__GRAPH_DATA__", json.dumps(graph_data, indent=2, default=str)
    )


def get_type(n: Node) -> str:
    if isinstance(n.data, SystemPrompt):
        return "system_prompt"
    elif isinstance(n.data, ToolDefinitions):
        return "tool_definitions"
    elif isinstance(n.data, ToolExecution):
        return "tool_exec"
    elif isinstance(n.data, StopReason):
        return "stop_reason"
    elif isinstance(n.data, SubGraph):
        return "sub_graph"
    elif isinstance(n.data, Message):
        return f"message_{n.data.role.value}"
    return "unknown"


def get_label(n: Node) -> str:
    if isinstance(n.data, SystemPrompt):
        return "SYSTEM"
    elif isinstance(n.data, ToolDefinitions):
        return f"TOOLS ({len(n.data.tools)})"
    elif isinstance(n.data, ToolExecution):
        return f"⚡{n.data.tool_name}"
    elif isinstance(n.data, StopReason):
        return f"◉ {n.data.reason}"
    elif isinstance(n.data, SubGraph):
        depth_str = f"[d={n.data.depth}]" if n.data.depth > 0 else ""
        return f"🔄{n.data.tool_name}{depth_str}"
    elif isinstance(n.data, Message):
        role = "USER" if n.data.role == Role.USER else "ASST"
        content = n.data.content
        if isinstance(content, str):
            preview = content[:20]
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, TextContent):
                    parts.append(block.text[:15])
                elif isinstance(block, ToolUseContent):
                    parts.append(f"→{block.name}")
                else:
                    parts.append("←result")
            preview = " ".join(parts)[:20]
        else:
            preview = str(content)[:20]
        return f"{role}: {preview}"
    return "?"


def serialize_data(data: Any) -> dict[str, Any]:
    if hasattr(data, "to_dict"):
        result: dict[str, Any] = data.to_dict()
        return result
    return {"raw": str(data)}


def view_file(filepath: str, output: str | None = None) -> None:
    """Load graph and generate interactive HTML."""
    heads, meta = Node.load_graph(filepath)

    html = to_html(heads[0], meta)

    out_path = Path(output) if output else Path(filepath).with_suffix(".html")
    out_path.write_text(html)
    print(f"Wrote: {out_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python viewer.py <graph.json> [output.html]")
        sys.exit(1)

    filepath = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    view_file(filepath, output)


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Node Graph Viewer</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script src="https://unpkg.com/@dagrejs/dagre@1.0.4/dist/dagre.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; flex-direction: column; height: 100vh; background: #1a1a2e; color: #eee; }
#breadcrumb { background: #0f0f23; border-bottom: 1px solid #333; padding: 10px 20px; display: flex; align-items: center; gap: 8px; min-height: 44px; }
#breadcrumb .crumb { color: #888; cursor: pointer; padding: 4px 10px; border-radius: 4px; font-size: 13px; transition: all 0.2s; }
#breadcrumb .crumb:hover { background: #1a1a3e; color: #fff; }
#breadcrumb .crumb.active { background: #ff9800; color: #000; font-weight: 600; cursor: default; }
#breadcrumb .sep { color: #555; font-size: 14px; }
#breadcrumb .hint { color: #666; font-size: 11px; margin-left: auto; font-style: italic; }
#main-container { flex: 1; display: flex; overflow: hidden; }
#graph { flex: 1; overflow: hidden; }
#detail { width: 400px; background: #16213e; border-left: 1px solid #333; overflow-y: auto; display: none; }
#detail.active { display: block; }
#detail-header { padding: 15px; background: #1a1a2e; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }
#detail-header h3 { font-size: 14px; color: #888; }
#detail-close { background: none; border: none; color: #888; font-size: 20px; cursor: pointer; }
#detail-close:hover { color: #fff; }
#detail-content { padding: 15px; }
.detail-section { margin-bottom: 20px; }
.detail-section h4 { font-size: 11px; text-transform: uppercase; color: #666; margin-bottom: 8px; letter-spacing: 1px; }
.detail-section pre { background: #0f0f23; padding: 12px; border-radius: 6px; font-size: 12px; overflow-x: auto; white-space: pre-wrap; word-break: break-word; }
.detail-section .value { background: #0f0f23; padding: 8px 12px; border-radius: 6px; font-size: 13px; }
.tools-list { background: #0f0f23; border-radius: 6px; padding: 8px 0; }
.tool-item { padding: 8px 12px; border-bottom: 1px solid #1a1a2e; }
.tool-item:last-child { border-bottom: none; }
.tool-name { color: #d9944a; font-weight: 600; font-size: 14px; margin-bottom: 6px; }
.tool-desc { color: #ccc; font-size: 12px; margin-top: 4px; line-height: 1.5; white-space: pre-wrap; }
.tool-schema { margin-top: 10px; }
.tool-schema .schema-label { color: #666; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
.tool-schema pre { margin-top: 4px; font-size: 11px; background: #0a0a1a; padding: 10px; border-radius: 4px; }
.content-block { margin-bottom: 12px; border-radius: 6px; overflow: hidden; }
.content-block:last-child { margin-bottom: 0; }
.block-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; padding: 6px 10px; font-weight: 600; }
.block-content { padding: 10px; font-size: 12px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
.content-text { background: #0f0f23; }
.content-text .block-label { background: #1a1a3e; color: #888; }
.content-text .block-content { color: #eee; }
.content-thinking { background: #1a1a0f; border: 1px solid #3d3d1a; }
.content-thinking .block-label { background: #2d2d1a; color: #d4c057; }
.content-thinking .block-content { color: #c9b84c; font-style: italic; }
.content-tool-use { background: #0f1a1a; border: 1px solid #1a3d3d; }
.content-tool-use .block-label { background: #1a2d2d; color: #57b4d4; }
.content-tool-use .block-content { color: #4cc9c9; }
.content-tool-result { background: #0f1a0f; border: 1px solid #1a3d1a; }
.content-tool-result .block-label { background: #1a2d1a; color: #57d457; }
.content-tool-result .block-content { color: #4cc94c; }
.content-system { background: #0f1a2a; border: 1px solid #1a3d5d; }
.content-system .block-label { background: #1a2d4d; color: #5794d4; }
.content-system .block-content { color: #8ab4e8; }
.content-user { background: #0f2a1a; border: 1px solid #1a5d3d; }
.content-user .block-label { background: #1a4d2d; color: #57d494; }
.content-user .block-content { color: #8ae8b4; }
.content-tool-exec { background: #2a1a0f; border: 1px solid #5d3d1a; }
.content-tool-exec .block-label { background: #4d2d1a; color: #d49457; }
.content-tool-exec .block-content { color: #e8b48a; }
.node { cursor: pointer; }
.node rect { stroke-width: 2px; }
.node text { font-size: 11px; fill: #333; }
.node.selected rect { stroke: #ffd700 !important; stroke-width: 3px; }
.edge { fill: none; stroke: #555; stroke-width: 2px; }
.edge-arrow { fill: #555; }
.type-system_prompt rect { fill: #e6f3ff; stroke: #4a90d9; }
.type-tool_definitions rect { fill: #fff3e6; stroke: #d9944a; }
.type-message_user rect { fill: #e6ffe6; stroke: #4ad94a; }
.type-message_assistant rect { fill: #f5f5f5; stroke: #888; }
.type-tool_exec rect { fill: #fff0e6; stroke: #d97a4a; }
.type-dict rect { fill: #ffe6f3; stroke: #d94a90; }
.type-stop_reason rect { fill: #ff5555; stroke: #ff0000; }
.type-stop_reason text { fill: #fff !important; font-weight: bold; }
/* SubGraph nodes: dashed border, clickable indicator */
.type-sub_graph rect { fill: #fff3e0; stroke: #ff9800; stroke-dasharray: 5,3; }
.type-sub_graph text { fill: #e65100 !important; font-weight: bold; }
.type-sub_graph:hover rect { stroke-width: 3px; }
/* Zoom button styles */
.zoom-btn { background: #ff9800; color: #000; border: none; padding: 10px 20px; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; width: 100%; margin-top: 10px; transition: all 0.2s; }
.zoom-btn:hover { background: #ffa726; transform: translateY(-1px); }
.zoom-btn.zoom-out { background: #666; color: #fff; }
.zoom-btn.zoom-out:hover { background: #888; }
svg { width: 100%; height: 100%; }
</style>
</head>
<body>
<div id="breadcrumb">
  <span class="crumb active">Main</span>
  <span class="hint">Click SubGraph nodes for zoom button</span>
</div>
<div id="main-container">
<div id="graph"></div>
<div id="detail">
  <div id="detail-header">
    <h3>NODE DETAILS</h3>
    <button id="detail-close">&times;</button>
  </div>
  <div id="detail-content"></div>
</div>
</div>

<script>
const graphData = __GRAPH_DATA__;

// Build tool_use_id -> tool_name mapping from all nodes
const toolNameMap = {};
graphData.nodes.forEach(n => {
  if (n.data && n.data.content && Array.isArray(n.data.content)) {
    n.data.content.forEach(block => {
      if (block.type === 'tool_use' && block.id && block.name) {
        toolNameMap[block.id] = block.name;
      }
    });
  }
});

const container = document.getElementById('graph');
const width = container.clientWidth;
const height = container.clientHeight;

const svg = d3.select('#graph').append('svg');
const g = svg.append('g');

// Arrow marker
svg.append('defs').append('marker')
  .attr('id', 'arrow')
  .attr('viewBox', '0 -5 10 10')
  .attr('refX', 8)
  .attr('refY', 0)
  .attr('markerWidth', 6)
  .attr('markerHeight', 6)
  .attr('orient', 'auto')
  .append('path')
  .attr('d', 'M0,-5L10,0L0,5')
  .attr('class', 'edge-arrow');

// Zoom
svg.call(d3.zoom().on('zoom', (e) => g.attr('transform', e.transform)));

// Build dagre graph
const dagreGraph = new dagre.graphlib.Graph();
dagreGraph.setGraph({ rankdir: 'TB', nodesep: 50, ranksep: 60, marginx: 20, marginy: 20 });
dagreGraph.setDefaultEdgeLabel(() => ({}));

const nodeWidth = 140, nodeHeight = 40;

// Add nodes
graphData.nodes.forEach(n => {
  dagreGraph.setNode(n.id, { width: nodeWidth, height: nodeHeight, data: n });
});

// Add edges
graphData.edges.forEach(e => {
  dagreGraph.setEdge(e.source, e.target);
});

// Layout
dagre.layout(dagreGraph);

// Get positions
const nodeMap = new Map();
dagreGraph.nodes().forEach(id => {
  const node = dagreGraph.node(id);
  node.data.x = node.x;
  node.data.y = node.y;
  nodeMap.set(id, node.data);
});

// Compute bounds and set viewBox
const xs = graphData.nodes.map(n => n.x);
const ys = graphData.nodes.map(n => n.y);
const minX = Math.min(...xs) - nodeWidth;
const maxX = Math.max(...xs) + nodeWidth;
const minY = Math.min(...ys) - nodeHeight;
const maxY = Math.max(...ys) + nodeHeight;
svg.attr('viewBox', `${minX} ${minY} ${maxX - minX} ${maxY - minY}`);

// Draw edges with curves
const edges = g.selectAll('.edge')
  .data(graphData.edges)
  .join('path')
  .attr('class', 'edge')
  .attr('marker-end', 'url(#arrow)')
  .attr('d', d => {
    const source = nodeMap.get(d.source);
    const target = nodeMap.get(d.target);
    const sy = source.y + nodeHeight/2;
    const ty = target.y - nodeHeight/2;
    const midY = (sy + ty) / 2;
    return `M${source.x},${sy} C${source.x},${midY} ${target.x},${midY} ${target.x},${ty}`;
  });

// Draw nodes
const nodes = g.selectAll('.node')
  .data(graphData.nodes)
  .join('g')
  .attr('class', d => `node type-${d.type}`)
  .attr('transform', d => `translate(${d.x - nodeWidth/2},${d.y - nodeHeight/2})`)
  .on('click', (e, d) => showDetail(d));

nodes.append('rect')
  .attr('width', nodeWidth)
  .attr('height', nodeHeight)
  .attr('rx', 6);

nodes.append('text')
  .attr('x', nodeWidth / 2)
  .attr('y', nodeHeight / 2 + 4)
  .attr('text-anchor', 'middle')
  .text(d => d.label.length > 18 ? d.label.slice(0, 16) + '...' : d.label);

// Detail panel
let selected = null;

function showDetail(node) {
  if (selected) d3.selectAll('.node').classed('selected', false);
  selected = node;
  d3.selectAll('.node').classed('selected', d => d.id === node.id);

  const detail = document.getElementById('detail');
  detail.classList.add('active');

  const content = document.getElementById('detail-content');
  content.innerHTML = buildDetailHTML(node);
}

function buildDetailHTML(node) {
  let html = '';

  html += section('Type', `<div class="value">${node.type}</div>`);
  html += section('ID', `<div class="value">${node.id}</div>`);

  // Handle different node types
  if (node.type === 'system_prompt') {
    html += section('Content', formatSystemPrompt(node.data.content));
  } else if (node.type === 'message_user') {
    html += section('Content', formatUserMessage(node.data.content));
  } else if (node.data.content !== undefined) {
    html += section('Content', formatContent(node.data.content));
  }

  // Tool execution node
  if (node.type === 'tool_exec') {
    html += section('Tool Execution', formatToolExec(node.data));
  }

  // SubGraph node
  if (node.type === 'sub_graph') {
    const nodeCount = node.data.nodes ? Object.keys(node.data.nodes).length : 0;
    html += section('Sub-Agent', `
      <div class="content-block content-tool-exec">
        <div class="block-label">🔄 ${escapeHtml(node.data.tool_name)} (depth: ${node.data.depth || 0})</div>
        <div class="block-content">${escapeHtml(node.data.summary || '(no summary)')}</div>
      </div>
      ${nodeCount > 0 ? `<button class="zoom-btn" onclick="zoomIntoSubGraph()">🔍 Zoom Into Sub-Agent (${nodeCount} nodes)</button>` : ''}
    `);
    if (node.data.system_prompt) {
      html += section('Sub-Agent System Prompt', `<div class="content-block content-system"><div class="block-content">${escapeHtml(node.data.system_prompt)}</div></div>`);
    }
  }

  // Stop reason node
  if (node.type === 'stop_reason') {
    html += section('Stop Reason', `<div class="value" style="color: #ff5555; font-weight: bold;">${escapeHtml(node.data.reason)}</div>`);
    if (node.data.usage) {
      const u = node.data.usage;
      const total_input = u.input_tokens || 0;
      const total_output = u.output_tokens || 0;
      const cached = u.cache_read_input_tokens || 0;
      const cache_write = u.cache_creation_input_tokens || 0;
      html += section('Total Usage', `
        <div class="value">
          <div><strong>Input:</strong> ${total_input.toLocaleString()} tokens ${cached > 0 ? `(${cached.toLocaleString()} cached)` : ''}</div>
          <div><strong>Output:</strong> ${total_output.toLocaleString()} tokens</div>
          ${cache_write > 0 ? `<div><strong>Cache writes:</strong> ${cache_write.toLocaleString()} tokens</div>` : ''}
          <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #333;"><strong>Total:</strong> ${(total_input + total_output).toLocaleString()} tokens</div>
        </div>
      `);
    }
  }

  if (node.data.tools) {
    const toolsHtml = node.data.tools.map(t => {
      let itemHtml = `<div class="tool-item">`;
      itemHtml += `<div class="tool-name">${escapeHtml(t.name)}</div>`;
      if (t.description) {
        itemHtml += `<div class="tool-desc">${escapeHtml(t.description)}</div>`;
      }
      if (t.input_schema) {
        itemHtml += `<div class="tool-schema"><span class="schema-label">Input Schema:</span><pre>${escapeHtml(JSON.stringify(t.input_schema, null, 2))}</pre></div>`;
      }
      itemHtml += `</div>`;
      return itemHtml;
    }).join('');
    html += section('Tools', `<div class="tools-list">${toolsHtml}</div>`);
  }

  if (Object.keys(node.metadata).length > 0) {
    html += section('Metadata', `<pre>${escapeHtml(JSON.stringify(node.metadata, null, 2))}</pre>`);
  }

  html += section('Raw Data', `<pre>${escapeHtml(JSON.stringify(node.data, null, 2))}</pre>`);

  return html;
}

function section(title, content) {
  return `<div class="detail-section"><h4>${title}</h4>${content}</div>`;
}

function formatSystemPrompt(content) {
  return `<div class="content-block content-system"><div class="block-label">System Prompt</div><div class="block-content">${escapeHtml(content)}</div></div>`;
}

function formatUserMessage(content) {
  if (typeof content === 'string') {
    return `<div class="content-block content-user"><div class="block-label">User</div><div class="block-content">${escapeHtml(content)}</div></div>`;
  }
  // If array (e.g., tool results), format each block
  if (Array.isArray(content)) {
    return content.map(block => {
      if (block.type === 'tool_result') {
        const toolName = toolNameMap[block.tool_use_id] || '';
        const label = toolName ? `← ${toolName}` : 'Tool Result';
        const resultContent = extractText(block.content);
        return `<div class="content-block content-tool-result"><div class="block-label">${escapeHtml(label)}</div><div class="block-content">${escapeHtml(resultContent)}</div></div>`;
      }
      return `<div class="content-block content-user"><div class="block-label">User</div><div class="block-content">${escapeHtml(JSON.stringify(block, null, 2))}</div></div>`;
    }).join('');
  }
  return `<pre>${escapeHtml(JSON.stringify(content, null, 2))}</pre>`;
}

function formatToolExec(data) {
  const resultText = extractText(data.result);
  return `<div class="content-block content-tool-exec">
    <div class="block-label">⚡ ${escapeHtml(data.tool_name)}</div>
    <div class="block-content">${escapeHtml(resultText)}</div>
  </div>`;
}

function formatContent(content) {
  if (typeof content === 'string') {
    return `<div class="content-block content-text"><div class="block-label">Text</div><div class="block-content">${escapeHtml(content)}</div></div>`;
  }
  if (Array.isArray(content)) {
    return content.map(block => {
      if (block.type === 'text') {
        return `<div class="content-block content-text"><div class="block-label">Text</div><div class="block-content">${escapeHtml(block.text)}</div></div>`;
      }
      if (block.type === 'thinking') {
        return `<div class="content-block content-thinking"><div class="block-label">Thinking</div><div class="block-content">${escapeHtml(block.thinking)}</div></div>`;
      }
      if (block.type === 'tool_use') {
        return `<div class="content-block content-tool-use"><div class="block-label">Tool: ${escapeHtml(block.name)}</div><pre class="block-content">${escapeHtml(JSON.stringify(block.input, null, 2))}</pre></div>`;
      }
      if (block.type === 'tool_result') {
        const toolName = toolNameMap[block.tool_use_id] || '';
        const label = toolName ? `← ${toolName}` : 'Tool Result';
        const resultContent = extractText(block.content);
        return `<div class="content-block content-tool-result"><div class="block-label">${escapeHtml(label)}</div><div class="block-content">${escapeHtml(resultContent)}</div></div>`;
      }
      return `<div class="content-block"><pre class="block-content">${escapeHtml(JSON.stringify(block, null, 2))}</pre></div>`;
    }).join('');
  }
  return `<pre>${escapeHtml(JSON.stringify(content, null, 2))}</pre>`;
}

function extractText(content) {
  // Extract text from string or list[TextContent]
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map(c => c.text || JSON.stringify(c)).join('\\n');
  }
  return JSON.stringify(content, null, 2);
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

document.getElementById('detail-close').onclick = () => {
  document.getElementById('detail').classList.remove('active');
  if (selected) d3.selectAll('.node').classed('selected', false);
  selected = null;
};

// Zoom into selected SubGraph node
function zoomIntoSubGraph() {
  if (selected && selected.type === 'sub_graph') {
    enterSubGraph(selected);
    // Close detail panel after zooming
    document.getElementById('detail').classList.remove('active');
    d3.selectAll('.node').classed('selected', false);
    selected = null;
  }
}

// Show meta info
console.log('Graph loaded:', graphData.meta);

// =============================================================================
// SubGraph Navigation System
// =============================================================================

// Stack of graph levels for navigation
const graphStack = [{
  name: 'Main',
  nodes: graphData.nodes,
  edges: graphData.edges,
  meta: graphData.meta
}];
let currentDepth = 0;

// Parse SubGraph internal nodes into viewer format
function parseSubGraphNodes(subGraphData) {
  const nodes = [];
  const nodeMap = subGraphData.nodes || {};

  for (const [id, nodeData] of Object.entries(nodeMap)) {
    if (!nodeData || typeof nodeData !== 'object') continue;

    const data = nodeData.data || {};
    const nodeType = getTypeFromData(data);

    nodes.push({
      id: id,
      type: nodeType,
      label: getLabelFromData(data, nodeType),
      data: data,
      metadata: nodeData.metadata || {},
      timestamp: nodeData.timestamp || 0,
      parent_count: (nodeData.parent_ids || []).length,
      parent_ids: nodeData.parent_ids || []
    });
  }

  return nodes;
}

// Parse SubGraph edges from node parent_ids
function parseSubGraphEdges(subGraphData) {
  const edges = [];
  const nodeMap = subGraphData.nodes || {};

  for (const [id, nodeData] of Object.entries(nodeMap)) {
    if (!nodeData || typeof nodeData !== 'object') continue;

    const parentIds = nodeData.parent_ids || [];
    for (const parentId of parentIds) {
      edges.push({ source: parentId, target: id });
    }
  }

  return edges;
}

// Determine node type from serialized data
function getTypeFromData(data) {
  if (!data || typeof data !== 'object') return 'unknown';

  const type = data.type;
  if (type === 'system_prompt') return 'system_prompt';
  if (type === 'tool_definitions') return 'tool_definitions';
  if (type === 'tool_execution') return 'tool_exec';
  if (type === 'stop_reason') return 'stop_reason';
  if (type === 'sub_graph') return 'sub_graph';

  // Check for message by role
  if (data.role === 'user') return 'message_user';
  if (data.role === 'assistant') return 'message_assistant';

  return 'unknown';
}

// Generate label from serialized data
function getLabelFromData(data, nodeType) {
  if (!data) return '?';

  if (nodeType === 'system_prompt') return 'SYSTEM';
  if (nodeType === 'tool_definitions') {
    const tools = data.tools || [];
    return `TOOLS (${tools.length})`;
  }
  if (nodeType === 'tool_exec') {
    return `⚡${data.tool_name || 'exec'}`;
  }
  if (nodeType === 'stop_reason') {
    return `◉ ${data.reason || 'stop'}`;
  }
  if (nodeType === 'sub_graph') {
    const depth = data.depth || 0;
    const depthStr = depth > 0 ? `[d=${depth}]` : '';
    return `🔄${data.tool_name || 'SubAgent'}${depthStr}`;
  }
  if (nodeType === 'message_user' || nodeType === 'message_assistant') {
    const role = nodeType === 'message_user' ? 'USER' : 'ASST';
    const content = data.content;
    let preview = '';
    if (typeof content === 'string') {
      preview = content.slice(0, 20);
    } else if (Array.isArray(content) && content.length > 0) {
      const first = content[0];
      if (first.type === 'text') preview = (first.text || '').slice(0, 15);
      else if (first.type === 'tool_use') preview = `→${first.name || 'tool'}`;
      else if (first.type === 'tool_result') preview = '←result';
      else preview = '...';
    }
    return `${role}: ${preview}`;
  }

  return '?';
}

// Enter a SubGraph (drill down)
function enterSubGraph(subGraphNode) {
  if (!subGraphNode.data || !subGraphNode.data.nodes) {
    console.warn('SubGraph has no nodes to explore');
    return;
  }

  const subNodes = parseSubGraphNodes(subGraphNode.data);
  const subEdges = parseSubGraphEdges(subGraphNode.data);

  if (subNodes.length === 0) {
    console.warn('SubGraph has no parseable nodes');
    return;
  }

  // Push to stack
  graphStack.push({
    name: subGraphNode.data.tool_name || 'SubAgent',
    nodes: subNodes,
    edges: subEdges,
    systemPrompt: subGraphNode.data.system_prompt,
    summary: subGraphNode.data.summary,
    depth: subGraphNode.data.depth || 0
  });
  currentDepth++;

  // Re-render
  renderCurrentGraph();
  updateBreadcrumb();
}

// Navigate to a specific depth in the stack
function navigateToDepth(depth) {
  if (depth < 0 || depth >= graphStack.length) return;

  // Pop stack until we reach target depth
  while (graphStack.length > depth + 1) {
    graphStack.pop();
  }
  currentDepth = depth;

  // Re-render
  renderCurrentGraph();
  updateBreadcrumb();
}

// Update breadcrumb UI
function updateBreadcrumb() {
  const breadcrumb = document.getElementById('breadcrumb');
  let html = '';

  graphStack.forEach((level, i) => {
    if (i > 0) html += '<span class="sep">›</span>';
    const activeClass = i === currentDepth ? ' active' : '';
    const onclick = i === currentDepth ? '' : `onclick="navigateToDepth(${i})"`;
    html += `<span class="crumb${activeClass}" ${onclick}>${escapeHtml(level.name)}</span>`;
  });

  // Add hint for SubGraph exploration
  const currentLevel = graphStack[currentDepth];
  const hasSubGraphs = currentLevel.nodes.some(n => n.type === 'sub_graph');
  if (hasSubGraphs) {
    html += '<span class="hint">Click SubGraph nodes for zoom button</span>';
  } else if (currentDepth > 0) {
    html += '<span class="hint">Click breadcrumb to navigate back</span>';
  }

  breadcrumb.innerHTML = html;
}

// Render the current graph level
function renderCurrentGraph() {
  const currentLevel = graphStack[currentDepth];
  const nodesData = currentLevel.nodes;
  const edgesData = currentLevel.edges;

  // Clear existing graph
  g.selectAll('*').remove();

  // Rebuild tool_use_id -> tool_name mapping for current level
  const localToolNameMap = {};
  nodesData.forEach(n => {
    if (n.data && n.data.content && Array.isArray(n.data.content)) {
      n.data.content.forEach(block => {
        if (block.type === 'tool_use' && block.id && block.name) {
          localToolNameMap[block.id] = block.name;
        }
      });
    }
  });
  // Merge with global map
  Object.assign(toolNameMap, localToolNameMap);

  // Build dagre graph
  const newDagreGraph = new dagre.graphlib.Graph();
  newDagreGraph.setGraph({ rankdir: 'TB', nodesep: 50, ranksep: 60, marginx: 20, marginy: 20 });
  newDagreGraph.setDefaultEdgeLabel(() => ({}));

  // Add nodes
  nodesData.forEach(n => {
    newDagreGraph.setNode(n.id, { width: nodeWidth, height: nodeHeight, data: n });
  });

  // Add edges
  edgesData.forEach(e => {
    newDagreGraph.setEdge(e.source, e.target);
  });

  // Layout
  dagre.layout(newDagreGraph);

  // Get positions
  const newNodeMap = new Map();
  newDagreGraph.nodes().forEach(id => {
    const node = newDagreGraph.node(id);
    if (node && node.data) {
      node.data.x = node.x;
      node.data.y = node.y;
      newNodeMap.set(id, node.data);
    }
  });

  // Compute bounds and set viewBox
  const xs = nodesData.map(n => n.x).filter(x => x !== undefined);
  const ys = nodesData.map(n => n.y).filter(y => y !== undefined);

  if (xs.length === 0 || ys.length === 0) {
    // No positioned nodes
    svg.attr('viewBox', '0 0 400 300');
    return;
  }

  const minX = Math.min(...xs) - nodeWidth;
  const maxX = Math.max(...xs) + nodeWidth;
  const minY = Math.min(...ys) - nodeHeight;
  const maxY = Math.max(...ys) + nodeHeight;
  svg.attr('viewBox', `${minX} ${minY} ${maxX - minX} ${maxY - minY}`);

  // Draw edges with curves
  g.selectAll('.edge')
    .data(edgesData.filter(d => newNodeMap.has(d.source) && newNodeMap.has(d.target)))
    .join('path')
    .attr('class', 'edge')
    .attr('marker-end', 'url(#arrow)')
    .attr('d', d => {
      const source = newNodeMap.get(d.source);
      const target = newNodeMap.get(d.target);
      const sy = source.y + nodeHeight/2;
      const ty = target.y - nodeHeight/2;
      const midY = (sy + ty) / 2;
      return `M${source.x},${sy} C${source.x},${midY} ${target.x},${midY} ${target.x},${ty}`;
    });

  // Draw nodes
  const newNodes = g.selectAll('.node')
    .data(nodesData.filter(n => n.x !== undefined))
    .join('g')
    .attr('class', d => `node type-${d.type}`)
    .attr('transform', d => `translate(${d.x - nodeWidth/2},${d.y - nodeHeight/2})`)
    .on('click', (e, d) => showDetail(d));

  newNodes.append('rect')
    .attr('width', nodeWidth)
    .attr('height', nodeHeight)
    .attr('rx', 6);

  newNodes.append('text')
    .attr('x', nodeWidth / 2)
    .attr('y', nodeHeight / 2 + 4)
    .attr('text-anchor', 'middle')
    .text(d => d.label.length > 18 ? d.label.slice(0, 16) + '...' : d.label);
}

// Initialize breadcrumb
updateBreadcrumb();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
