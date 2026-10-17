import { useState } from 'react';
import { Descriptions, Input, Table, Tag, Tree } from 'antd';
import type { DataNode } from 'antd/es/tree';
import { Panel, JsonView } from '../../components';
import type { Architecture, StructureModule } from './types';

export default function StructureExplorer({ data }: { data: Architecture }) {
  const [selected, setSelected] = useState('');
  const [search, setSearch] = useState('');
  const runtime = data.runtime.loaded_config_matches && data.runtime.modules;
  const structure = runtime || data.structure;
  const rows = structure?.modules ?? [];
  const chosen = rows.find((row) => row.path === selected);
  const nodes = new Map<string, DataNode>(
    rows.map((row) => [
      row.path,
      {
        key: row.path || '@root',
        title: row.path
          ? row.path.split('.').at(-1)!
          : runtime
            ? (data.runtime.model_class ?? 'Model')
            : '权重结构',
        children: [],
      },
    ]),
  );
  for (const row of rows) {
    if (row.parent !== null) nodes.get(row.parent)?.children?.push(nodes.get(row.path)!);
  }
  const weights = data.tensors.filter((row) => !selected || row.name.startsWith(selected + '.'));
  return (
    <Panel
      title="通用模块与权重结构"
      subtitle={
        runtime
          ? '来自当前驻留模型的真实 nn.Module；树线表示包含关系。'
          : '按实际权重路径组织，不依赖模型族；树线表示包含关系，不表示 forward 依赖。'
      }
    >
      <Tag>{runtime ? '运行时模块树' : '制品权重结构'}</Tag>
      <Tag>{rows.length} 个结构节点</Tag>
      <Input.Search
        aria-label="筛选模块路径"
        placeholder="筛选模块路径 / 类名"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        allowClear
      />
      {search ? (
        <Table<StructureModule>
          size="small"
          rowKey="path"
          pagination={{ pageSize: 8 }}
          dataSource={rows.filter((row) =>
            (row.path + ' ' + (row.class ?? '')).toLowerCase().includes(search.toLowerCase()),
          )}
          onRow={(row) => ({ onClick: () => setSelected(row.path) })}
          columns={[
            {
              title: '模块路径',
              dataIndex: 'path',
              render: (value) => <code>{value || '(root)'}</code>,
            },
            { title: '实际类', dataIndex: 'class' },
            { title: '语义标注', dataIndex: 'role' },
          ]}
        />
      ) : (
        <Tree
          aria-label="通用模块树"
          height={320}
          treeData={nodes.has('') ? [nodes.get('')!] : []}
          defaultExpandedKeys={['@root', 'model', 'model.layers', 'transformer', 'transformer.h']}
          selectedKeys={[selected || '@root']}
          onSelect={(keys) => setSelected(keys[0] === '@root' ? '' : String(keys[0] ?? ''))}
          showLine
          blockNode
        />
      )}
      <Descriptions
        size="small"
        column={2}
        items={[
          { key: 'path', label: '所选路径', children: <code>{selected || '(root)'}</code> },
          {
            key: 'class',
            label: '实际模块类',
            children: chosen?.class ?? '制品路径；加载后可确认实际模块类',
          },
          { key: 'role', label: '语义标注', children: chosen?.role ?? 'root' },
          { key: 'weights', label: '关联制品 Tensor', children: weights.length },
        ]}
      />
      {selected && (
        <Table
          size="small"
          rowKey="name"
          pagination={{ pageSize: 6, showSizeChanger: false }}
          dataSource={weights}
          columns={[
            { title: 'Tensor', dataIndex: 'name', render: (value) => <code>{value}</code> },
            { title: 'Shape', dataIndex: 'shape', render: (value) => value.join(' × ') },
            { title: '存储 dtype', dataIndex: 'dtype' },
          ]}
        />
      )}
      <details>
        <summary>结构来源与当前模块元数据</summary>
        <JsonView
          value={{ source: structure?.source, status: structure?.status, module: chosen }}
        />
      </details>
    </Panel>
  );
}
