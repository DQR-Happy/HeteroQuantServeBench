import type { ArchitectureGraph, ArchitectureNode } from './types';

const W = 170;
const H = 64;
const position = (n: ArchitectureNode) => ({ x: 24 + n.column * 200, y: 40 + n.row * 126 });

export default function DependencyGraph({
  graph,
  selected,
  onSelect,
  identity,
}: {
  graph: ArchitectureGraph;
  selected?: string;
  onSelect: (node: ArchitectureNode) => void;
  identity: string;
}) {
  const nodes = new Map(graph.nodes.map((n) => [n.id, n]));
  const width = Math.max(...graph.nodes.map((n) => position(n).x)) + W + 30;
  const height = Math.max(...graph.nodes.map((n) => position(n).y)) + H + 40;
  return (
    <div
      className="architecture-canvas"
      tabIndex={0}
      aria-label="逻辑计算依赖图，可横向滚动，节点可用 Tab 选择"
    >
      <svg
        width={width}
        height={height}
        role="group"
        aria-label={`${identity} 计算依赖，箭头表示数据依赖，不表示实际 GPU 并行`}
      >
        <defs>
          <marker
            id={`arrow-${identity}`}
            markerWidth="7"
            markerHeight="7"
            refX="6"
            refY="3.5"
            orient="auto"
          >
            <path d="M0 0 L7 3.5 L0 7 z" fill="currentColor" />
          </marker>
        </defs>
        {graph.edges.map((edge, i) => {
          const from = position(nodes.get(edge.source)!);
          const to = position(nodes.get(edge.target)!);
          const x1 = from.x + W,
            y1 = from.y + H / 2,
            x2 = to.x,
            y2 = to.y + H / 2;
          const mid = (x1 + x2) / 2;
          return (
            <path
              key={i}
              className={`architecture-edge ${edge.kind}`}
              d={`M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`}
              markerEnd={`url(#arrow-${identity})`}
            />
          );
        })}
        {graph.nodes.map((node) => {
          const p = position(node);
          return (
            <g
              key={node.id}
              className={`architecture-node ${node.kind} ${selected === node.id ? 'selected' : ''}`}
              transform={`translate(${p.x},${p.y})`}
              role="button"
              tabIndex={0}
              aria-label={`${node.label}${node.module ? `，${node.module}` : ''}`}
              onClick={() => onSelect(node)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault();
                  onSelect(node);
                }
              }}
            >
              <rect width={W} height={H} rx="6" />
              <text x={W / 2} y="26" textAnchor="middle">
                {node.label}
              </text>
              <text className="architecture-node-caption" x={W / 2} y="47" textAnchor="middle">
                {node.kind === 'state'
                  ? '逻辑状态'
                  : node.kind === 'residual'
                    ? '恒等依赖'
                    : node.module
                      ? '模型模块'
                      : '数学操作'}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
