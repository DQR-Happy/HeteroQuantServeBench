import Button from '../../components/ActionButton';
import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Table, Tooltip } from 'antd';
import { Link } from 'react-router-dom';
import { api, fmt } from '../../api/client';
import type { Run } from '../../api/types';
import { QueryState, Stat } from '../../components';

export type BaselineResult = {
  status: 'baseline' | 'compared' | 'missing' | 'unavailable';
  reason?: string;
  reference_eligible: boolean;
  baseline?: {
    run_id: string;
    compute_dtype: string;
    perplexity: number | null;
    mean_nll: number;
    manifest_sha256: string;
  };
  quality?: {
    ppl_delta: number | null;
    ppl_relative_increase_percent: number | null;
    mean_nll_increase: number;
    task_accuracy_drop_percentage_points: number | null;
    documents?: { id: string; article: string; mean_nll_increase: number }[];
  };
};
const signed = (value: number | null | undefined, digits = 4) =>
  value == null || !Number.isFinite(value) ? '—' : `${value > 0 ? '+' : ''}${fmt(value, digits)}`;

function useBaselineQuery(run: Run) {
  return useQuery({
    queryKey: ['quality-baseline', run.id],
    queryFn: () => api<BaselineResult>(`/runs/${run.id}/quality-baseline`),
    enabled:
      run.state === 'completed' &&
      run.cleanup === 'succeeded' &&
      run.metrics.quality_evaluation?.status === 'complete',
    retry: false,
  });
}

export function QualityBaselineCell({ run }: { run: Run }) {
  const query = useBaselineQuery(run);
  const data = query.data;
  const delta = data?.quality?.ppl_relative_increase_percent;
  const label =
    data?.status === 'baseline'
      ? `基线 · ${data.baseline?.compute_dtype}`
      : data?.status === 'compared'
        ? delta == null
          ? '不可计算'
          : `${signed(delta, 2)}%`
        : data?.status === 'missing'
          ? '未建立基线'
          : query.isError || data?.status === 'unavailable'
            ? '基线不可核验'
            : run.state === 'completed'
              ? '读取中'
              : '尚无完整评分';
  return (
    <Tooltip
      title={
        data?.baseline
          ? `基线 PPL ${fmt(data.baseline.perplexity, 4)} · ${data.baseline.compute_dtype}；正值表示 PPL 上升（退化）`
          : '需要同模型、同版本和同协议的浮点评测'
      }
    >
      <Link to={`/runs/${run.id}?tab=quality`}>{label}</Link>
    </Tooltip>
  );
}

export default function QualityBaseline({ run }: { run: Run }) {
  const client = useQueryClient();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const query = useBaselineQuery(run);
  const result = query.data;
  const baseline = result?.baseline,
    delta = result?.quality;
  return (
    <section aria-label="模型浮点基线对照" className="section-gap">
      <h3>模型浮点基线对照</h3>
      <QueryState loading={query.isFetching} error={query.error} />
      {result?.status === 'missing' && (
        <Alert
          type="info"
          showIcon
          message="尚未建立同协议的模型浮点基线"
          description={
            <>
              先在<Link to="/precision">表示精度与量化</Link>选择 Floating point、默认 KV
              和框架原生算子，运行同一质量评测。系统会自动保存首个有效基线并关联量化记录；不同模型、权重版本、分词器、源码、设备、依赖版本或协议不会混用。
            </>
          }
        />
      )}
      {result?.status === 'unavailable' && (
        <Alert
          type="warning"
          message="已注册基线暂不可核验"
          description={`${result.reason}；保留候选实测值，不生成相对变化。请检查原件或显式重新指定合格浮点记录。`}
        />
      )}
      {baseline && delta && (
        <>
          <p>
            原始浮点权重 · <strong>{baseline.compute_dtype}</strong> · 默认 KV ·
            框架原生算子。基线记录：<Link to={`/runs/${baseline.run_id}`}>{baseline.run_id}</Link>
            {result?.status === 'baseline' ? '（本记录即当前基线）' : ''}
          </p>
          <div className="stats-grid">
            <Stat
              label="基线 PPL ↓"
              value={fmt(baseline.perplexity, 4)}
              hint="已封存的未量化模型实测值"
            />
            <Stat
              label="PPL 相对变化"
              value={
                delta.ppl_relative_increase_percent == null
                  ? '不可计算'
                  : `${signed(delta.ppl_relative_increase_percent, 2)}%`
              }
              hint="PPL 相对变化：正值为上升，负值为改善"
            />
            <Stat label="PPL 绝对变化" value={signed(delta.ppl_delta)} hint="当前 PPL − 基线 PPL" />
            <Stat
              label="平均 NLL 增量"
              value={signed(delta.mean_nll_increase, 6)}
              hint="当前平均 NLL − 基线平均 NLL"
            />
          </div>
          {delta.task_accuracy_drop_percentage_points != null && (
            <p>
              标准答案匹配率相对基线下降 {signed(delta.task_accuracy_drop_percentage_points, 2)}{' '}
              个百分点。
            </p>
          )}
          <p className="help-text">
            浮点基线按实际 FP16 / BF16 / FP32
            精度标注。只有同一权重来源、分词器、运行源码、Attention 和评分数据 /
            协议才关联；这里只报告变化，是否达标仍按预登记阈值判断。
          </p>
          {delta.documents && (
            <details>
              <summary>逐段相对基线变化</summary>
              <Table
                rowKey="id"
                size="small"
                pagination={false}
                scroll={{ x: 500 }}
                dataSource={delta.documents}
                columns={[
                  { title: '文章 / 样本', dataIndex: 'article' },
                  { title: '平均 NLL 增量', render: (_, s) => signed(s.mean_nll_increase, 4) },
                  {
                    title: 'PPL 相对变化',
                    render: (_, s) => `${signed(Math.expm1(s.mean_nll_increase) * 100, 2)}%`,
                  },
                ]}
              />
            </details>
          )}
        </>
      )}
      {result?.reference_eligible && result.status !== 'baseline' && (
        <Button
          aria-label="设为模型浮点基线"
          loading={busy}
          onClick={async () => {
            setBusy(true);
            setError('');
            try {
              await api('/precision/quality-baselines', { run_id: run.id });
              await client.invalidateQueries({ queryKey: ['quality-baseline'] });
            } catch (failure) {
              setError((failure as Error).message);
            } finally {
              setBusy(false);
            }
          }}
        >
          将本次浮点评测设为基线
        </Button>
      )}
      {error && <Alert type="error" message={error} />}
    </section>
  );
}
