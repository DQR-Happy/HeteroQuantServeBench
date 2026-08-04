import { Alert, Button, Progress, Space, Tag } from 'antd';
import { base, fmt, gib, terminal } from '../../api/client';
import type { Run } from '../../api/types';
import { JsonView, Panel, Stat } from '../../components';

export default function QuantizationRun({ run }: { run: Run }) {
  const progress = run.progress;
  const result = run.metrics.quantization;
  const total = progress?.total_tensors;
  return (
    <Panel
      title="量化转换任务"
      subtitle="任务在模型工作进程执行；候选制品与在线部署分离"
      extra={
        run.state === 'completed' && (
          <Button href={`${base}/quantization/artifacts/${encodeURIComponent(run.id)}/manifest`}>
            下载制品 Manifest
          </Button>
        )
      }
    >
      <Space wrap>
        <Tag>{result?.method ?? 'RTN'}</Tag>
        <Tag>{result?.execution_path ?? 'storage_only'}</Tag>
        <Tag>质量未评估</Tag>
      </Space>
      <Alert
        className="section-gap"
        type="info"
        showIcon
        message="完成转换只证明制品已生成；没有自动启用低比特推理。"
        description="后续仍需质量门、兼容执行 kernel 和未插桩的内存 / 速度对照验证。"
      />
      {progress && (
        <div className="section-gap">
          <h3>{terminal(run.state) ? '最后任务进度' : '实时任务进度'}</h3>
          <p className="small">
            阶段：{progress.stage ?? '—'} · 已完成 {fmt(progress.completed_tensors, 0)} /{' '}
            {fmt(total, 0)} 个权重张量
          </p>
          {total && progress.completed_tensors != null ? (
            <Progress
              percent={Math.min(
                100,
                Math.max(0, Math.round((progress.completed_tensors / total) * 100)),
              )}
              status={
                run.state === 'failed'
                  ? 'exception'
                  : run.state === 'completed'
                    ? 'success'
                    : 'normal'
              }
            />
          ) : null}
          {progress.tensor && <p className="small mono kernel-name">当前权重：{progress.tensor}</p>}
          {progress.rows_processed != null && (
            <p className="small muted">
              行进度：{fmt(progress.rows_processed, 0)} / {fmt(progress.total_rows, 0)}
            </p>
          )}
        </div>
      )}
      {result && (
        <>
          <div className="stats-grid">
            <Stat
              label="量化位宽"
              value={result.bits}
              unit="bit"
              hint={`group size: ${result.group_size ?? 'per-channel'}`}
            />
            <Stat
              label="Packed 权重"
              value={gib(result.bytes?.qvalues)}
              unit="GiB"
              hint="持久化数据，非驻留内存"
            />
            <Stat label="Scale" value={gib(result.bytes?.scales)} unit="GiB" hint="量化缩放信息" />
            <Stat
              label="制品总大小"
              value={gib(result.bytes?.total)}
              unit="GiB"
              hint="含数据、缩放与 manifest"
            />
          </div>
          <details>
            <summary>覆盖范围与量化误差</summary>
            <JsonView
              value={{
                coverage: result.coverage,
                weight_error: result.weight_error,
                source_identity_scope: result.source_identity_scope,
                native_deployment_available: result.native_deployment_available,
                resident_model_modified: result.resident_model_modified,
              }}
            />
          </details>
          <div className="help-text">
            {result.limitations?.map((limitation, index) => <p key={index}>{limitation}</p>)}
          </div>
        </>
      )}
      {run.result?.manifest != null && (
        <details>
          <summary>制品身份与完整 manifest</summary>
          <JsonView value={run.result.manifest} />
        </details>
      )}
    </Panel>
  );
}
