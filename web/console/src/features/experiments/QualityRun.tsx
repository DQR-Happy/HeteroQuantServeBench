import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Alert, Descriptions, Select, Table } from 'antd';
import { Link } from 'react-router-dom';
import { api, fmt, gib, terminal } from '../../api/client';
import type { Run } from '../../api/types';
import { Chart, JsonView, QueryState, Stat } from '../../components';
import type { Bundle, QualityResult } from './types';
import QualityBaseline from './QualityBaseline';

const scoreStates: Record<string, string> = {
  complete: '计分完成',
  partial: '仅有部分结果',
};

/** Read saved scores and sealed samples; never infer missing historical measurements. */
export default function QualityRun({ run }: { run: Run }) {
  const [selected, setSelected] = useState('');
  const query = useQuery({
    queryKey: ['experiment-bundle', run.id],
    queryFn: () => api<Bundle>(`/runs/${run.id}/experiment`),
    enabled: terminal(run.state),
    retry: false,
  });
  const data = query.data?.quality ?? run.metrics.quality_evaluation;
  const samples = query.data?.quality?.samples ?? [];
  const corpus = data?.score_mode === 'corpus_nll';
  const sample = samples.find((s) => s.id === selected) ?? samples[0];
  const protocol = data?.protocol;
  const receipt = query.data?.receipt ?? run.metrics.optimization_receipt;
  const precision = receipt?.precision_receipt;
  const memory = receipt?.proof?.after_cleanup_allocator as Run['metrics']['memory'] | undefined;
  const count = data?.document_count ?? (query.data?.quality ? samples.length : undefined);
  const complete = data?.status === 'complete';
  return (
    <>
      <div className="stats-grid">
        <Stat
          label="困惑度 PPL ↓"
          value={data?.perplexity == null ? undefined : fmt(data.perplexity, 4)}
          hint={
            data?.perplexity_overflow
              ? '数值溢出；请查看平均 NLL'
              : '同一模型、分词器和计分协议下越低越好'
          }
        />
        <Stat
          label="平均 NLL ↓"
          value={data?.mean_nll == null ? undefined : fmt(data.mean_nll, 6)}
          hint="真实后续 Token 的平均负对数似然"
        />
        <Stat label="计分 Token" value={data?.scored_tokens} />
        <Stat label="已完成样本" value={count} />
        <Stat
          label="评测任务总耗时"
          value={run.metrics.console_e2e_ms}
          unit="ms"
          hint="质量任务总耗时，包含排队与评分"
        />
      </div>
      <QualityBaseline run={run} />
      <p>
        评分状态：
        {data
          ? (scoreStates[data.status] ?? data.status)
          : terminal(run.state)
            ? '没有可用评分结果'
            : '等待任务完成与封存'}
        。计分完成后，可与浮点基线按预登记阈值对照。
      </p>
      <Alert
        type={complete ? 'info' : 'warning'}
        showIcon
        message="独立质量评测不生成聊天回答"
        description="本任务评估真实后续 Token 的预测概率，未采集 TTFT、生成 Token/s、Decode 速度、GPU profiler 或 OS 页级详录。这些指标在此不适用或未采集，不会填成 0；性能和机制需要另行运行对应任务。"
      />
      <QueryState loading={terminal(run.state) && query.isPending} error={query.error} />
      {query.error && <p>上方保留运行记录中的评分摘要；封存原件读取失败，逐样本证据暂不可核验。</p>}
      <Descriptions title="评分协议与实际配置" size="small" column={{ xs: 1, sm: 2 }} bordered>
        <Descriptions.Item label="数据集">
          {data?.benchmark_id ?? '自定义注册样本'}
        </Descriptions.Item>
        <Descriptions.Item label="计分方式">
          {protocol?.forward_mode === 'incremental_decode'
            ? '逐 Token teacher-forcing（经过当前 KV）'
            : String(protocol?.forward_mode ?? protocol?.name ?? '尚无协议')}
        </Descriptions.Item>
        <Descriptions.Item label="每条 Token 上限">
          {String(protocol?.max_tokens ?? '未记录')}
        </Descriptions.Item>
        <Descriptions.Item label="聚合方式">
          {corpus
            ? 'exp(总 NLL ÷ 总计分 Token)，段间重置、首 Token 不计分'
            : String(protocol?.name ?? '未记录')}
        </Descriptions.Item>
        <Descriptions.Item label="实际权重方案">
          {precision ? `${precision.method} / ${precision.compute_dtype}` : '见优化与质量凭据'}
        </Descriptions.Item>
        <Descriptions.Item label="实际 KV 配置">
          {precision?.kv
            ? precision.kv.bits == null
              ? '模型默认'
              : `${precision.kv.backend} INT${precision.kv.bits} · residual ${precision.kv.residual_length}`
            : '未记录'}
        </Descriptions.Item>
        <Descriptions.Item label="数据 SHA256" span="filled">
          <code style={{ overflowWrap: 'anywhere' }}>{data?.dataset_sha256 ?? '未记录'}</code>
        </Descriptions.Item>
      </Descriptions>
      {data?.task_accuracy != null ? (
        <p>标准答案匹配率：{fmt(data.task_accuracy * 100, 2)}%</p>
      ) : (
        <p className="help-text">
          本次使用语料 PPL 评分；问答准确率及中文、数学、代码、长上下文能力可选择相应任务集评测。
        </p>
      )}
      {samples.length > 0 && (
        <>
          <h3>逐样本评分</h3>
          <Table<QualityResult['samples'][number]>
            rowKey="id"
            size="small"
            pagination={false}
            scroll={{ x: 740 }}
            dataSource={samples}
            columns={[
              { title: '文章 / 样本', render: (_, s) => s.slice ?? s.id },
              {
                title: '输入 Token',
                dataIndex: 'input_tokens',
                render: (v: number | undefined) => fmt(v, 0),
              },
              ...(corpus
                ? [
                    {
                      title: '计分 Token',
                      dataIndex: 'scored_tokens',
                      render: (v: number | undefined) => fmt(v, 0),
                    },
                    {
                      title: '平均 NLL ↓',
                      render: (_: unknown, s: QualityResult['samples'][number]) =>
                        fmt(
                          s.nll_sum != null && s.scored_tokens
                            ? s.nll_sum / s.scored_tokens
                            : undefined,
                          4,
                        ),
                    },
                    {
                      title: 'PPL ↓',
                      render: (_: unknown, s: QualityResult['samples'][number]) =>
                        fmt(
                          s.nll_sum != null && s.scored_tokens
                            ? Math.exp(s.nll_sum / s.scored_tokens)
                            : undefined,
                          3,
                        ),
                    },
                  ]
                : [
                    { title: '续写 Token', dataIndex: 'output_tokens' },
                    {
                      title: '标准答案匹配',
                      render: (_: unknown, s: QualityResult['samples'][number]) =>
                        s.exact_match == null ? '无标签' : s.exact_match ? '匹配' : '不匹配',
                    },
                  ]),
            ]}
          />
          {sample?.token_nll && (
            <>
              <h3>逐 Token NLL</h3>
              <Select
                aria-label="评分样本"
                value={sample.id}
                onChange={setSelected}
                style={{ width: '100%', maxWidth: 540 }}
                options={samples.map((s) => ({ value: s.id, label: s.slice ?? s.id }))}
              />
              <Chart
                label="逐 Token 负对数似然"
                option={{
                  tooltip: { trigger: 'axis' },
                  grid: { left: 55, right: 25, bottom: 45, top: 20 },
                  xAxis: {
                    type: 'category',
                    name: '位置',
                    data: sample.token_nll.map((_, i) => i + 2),
                  },
                  yAxis: { type: 'value', name: 'NLL' },
                  series: [{ type: 'line', data: sample.token_nll, showSymbol: false }],
                }}
              />
              <p className="help-text">
                横轴为样本内目标 Token 位置，从第 2 个 Token 开始。NLL
                越高，模型给真实下文的概率越低。
              </p>
            </>
          )}
        </>
      )}
      {memory && (
        <>
          <h3>Worker 清理后内存快照</h3>
          <div className="stats-grid">
            <Stat label="清理后已分配" value={gib(memory.allocated_bytes)} unit="GiB" />
            <Stat label="清理后保留" value={gib(memory.reserved_bytes)} unit="GiB" />
            <Stat
              label="Worker 累计峰值"
              value={gib(memory.peak_allocated_bytes)}
              unit="GiB"
              hint="自上次 allocator 重置以来；不能当作本次评测的独立峰值"
            />
          </div>
          <p className="help-text">
            来源：执行凭据 after_cleanup_allocator，范围仅为 worker CUDA allocator
            快照。逐阶段内存、Cache 命中率和页错误可通过对应观测模式采集。
          </p>
        </>
      )}
      <p>
        <Link to="/precision">返回量化页进行基线与候选质量对照</Link>
      </p>
      {query.data?.quality && (
        <details>
          <summary>完整评分原件</summary>
          <JsonView value={query.data.quality} />
        </details>
      )}
    </>
  );
}
