import { Tag } from 'antd';
import { Link } from 'react-router-dom';
import { JsonView, Panel } from '../../components';
import type { Architecture } from './types';
export default function ArchitectureProvenance({ data }: { data: Architecture }) {
  const sizes = data.config;
  return (
    <Panel title="数据来源与采集范围" subtitle="制品静态结构、安装实现与当前加载状态分别记录">
      <p>
        静态语义示意图状态：
        <Tag>
          {data.status === 'verified_static'
            ? '制品关键结构与已核对实现匹配'
            : '暂无专用示意图；通用结构可用'}
        </Tag>
        实现：Transformers {data.implementation.version ?? '未识别'}
      </p>
      <p>
        模型族整理：<Tag>{data.semantic?.family ?? sizes.model_type}</Tag>
        依据配置声明与模块名称整理；执行结果和 kernel 调用见运行凭据。
      </p>
      <details>
        <summary>已覆盖的模型族与配置声明</summary>
        <JsonView value={data.semantic} />
      </details>
      <p>
        当前驻留模型的 config 与本次制品匹配：
        {data.runtime.loaded_config_matches == null
          ? '未加载，无法确认'
          : data.runtime.loaded_config_matches
            ? '是'
            : '否，请重新核对制品与加载记录'}
      </p>
      {data.runtime.load_run_id && (
        <Link to={`/runs/${data.runtime.load_run_id}`}>打开当前加载凭据 →</Link>
      )}
      <ul>
        {data.limitations.map((text) => (
          <li key={text}>{text}</li>
        ))}
      </ul>
      <details>
        <summary>配置、实现与分片 header 身份</summary>
        <JsonView
          value={{
            config: data.config,
            config_sha256: data.config_sha256,
            implementation: data.implementation,
            shards: data.shards,
            runtime: data.runtime,
            evidence_scope: data.evidence_scope,
            issues: data.issues,
          }}
        />
      </details>
    </Panel>
  );
}
