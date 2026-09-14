import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Button, Checkbox, InputNumber, Select, Space, Table, Tag } from 'antd';
import { Link, useSearchParams } from 'react-router-dom';
import { api, date } from '../api/client';
import type { Deployment } from '../api/types';
import { PageHead, Panel, QueryState, Status } from '../components';
import Results from '../features/benchmarks/Results';
import { applyKernelProfile } from '../features/kernels/lifecycle';
import { finished } from '../features/benchmarks/types';
import type {
  Benchmark,
  BenchmarkCatalogue,
  BenchmarkList,
  FixedCase,
} from '../features/benchmarks/types';

const phaseLabel: Record<string, string> = {
  starting: '准备计划',
  loading: '加载方案',
  unloading: '卸载方案',
  warmup: '用例预热',
  measurement: '性能测量与输出采样',
  diagnostic: '独立机制诊断',
  restoring: '恢复原部署',
  completed: '已完成',
  cancelled: '已取消',
  failed: '失败',
  interrupted: '服务中断',
};
export default function Benchmarks() {
  const client = useQueryClient();
  const [params, setParams] = useSearchParams();
  const identity = params.get('benchmark') ?? '';
  const [selectedDeployment, setSelectedDeployment] = useState(params.get('deployment') ?? '');
  const [baseline, setBaseline] = useState('framework-native');
  const [candidateDraft, setCandidates] = useState<string[] | null>(null);
  const [caseDraft, setCases] = useState<string[] | null>(null);
  const [repeats, setRepeats] = useState(1);
  const [budget, setBudget] = useState(64);
  const [diagnostics, setDiagnostics] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const switchController = useRef<AbortController | null>(null);
  useEffect(() => () => switchController.current?.abort(), []);
  const [error, setError] = useState('');
  const [configurationOpen, setConfigurationOpen] = useState(false);
  const deployments = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 3000,
  });
  const deployment =
    deployments.data?.items.find((item) => item.id === selectedDeployment) ??
    deployments.data?.items.find((item) => item.provider === 'pytorch');
  const catalog = useQuery({
    queryKey: ['benchmark-catalog', deployment?.id],
    queryFn: () =>
      api<BenchmarkCatalogue>(
        `/benchmark-catalog?deployment_id=${encodeURIComponent(deployment!.id)}`,
      ),
    enabled: !!deployment,
    refetchInterval: 3000,
  });
  const listing = useQuery({
    queryKey: ['benchmarks'],
    queryFn: () => api<BenchmarkList>('/benchmarks'),
    refetchInterval: 3000,
  });
  const result = useQuery({
    queryKey: ['benchmark', identity],
    queryFn: () => api<Benchmark>(`/benchmarks/${encodeURIComponent(identity)}`),
    enabled: !!identity,
    retry: false,
    refetchInterval: (query) =>
      query.state.data && finished(query.state.data.state) ? false : 1500,
  });
  const candidates = (
    candidateDraft ??
    catalog.data?.defaults.candidate_ids.filter((id) =>
      catalog.data?.schemes.some((scheme) => scheme.id === id && scheme.availability.loadable),
    ) ??
    []
  ).filter((item) => item !== baseline);
  const cases = caseDraft ?? catalog.data?.defaults.case_ids ?? [];
  const running = !!catalog.data?.active_id || (!!result.data && !finished(result.data.state));
  const launch = async () => {
    if (!deployment || !cases.length || !candidates.length || running || submitting) return;
    setSubmitting(true);
    setError('');
    const controller = new AbortController();
    switchController.current = controller;
    try {
      const owner = deployments.data?.items.find(
        (item) => item.id !== deployment.id && item.provider === 'pytorch' &&
          !['unloaded', 'failed'].includes(item.state),
      );
      if (owner) {
        const profile = catalog.data?.schemes.find((item) => item.id === baseline)?.kernel_profile;
        if (!profile) throw new Error('请先选择有效基线方案。');
        await applyKernelProfile(deployment, profile, controller.signal, () => {});
        void client.invalidateQueries({ queryKey: ['deployments'] });
      }
      controller.signal.throwIfAborted();
      const created = await api<Benchmark>('/benchmarks', {
        deployment_id: deployment.id,
        baseline_id: baseline,
        candidate_ids: candidates,
        case_ids: cases,
        repeats,
        output_tokens: budget,
        diagnostics,
      });
      client.setQueryData(['benchmark', created.id], created);
      if (controller.signal.aborted) return;
      setParams({ benchmark: created.id });
      setConfigurationOpen(false);
      void client.invalidateQueries({ queryKey: ['benchmark-catalog'] });
      void client.invalidateQueries({ queryKey: ['benchmarks'] });
    } catch (failure) {
      if (!controller.signal.aborted) setError((failure as Error).message);
    } finally {
      if (!controller.signal.aborted) {
        switchController.current = null;
        setSubmitting(false);
      }
    }
  };
  const cancel = async () => {
    try {
      const updated = await api<Benchmark>(`/benchmarks/${identity}/cancel`, {});
      client.setQueryData(['benchmark', identity], updated);
    } catch (failure) {
      setError((failure as Error).message);
    }
  };
  const options = catalog.data?.schemes.map((scheme) => ({
    value: scheme.id,
    label: scheme.name,
    disabled: !scheme.availability.loadable,
    title: scheme.availability.reason ?? scheme.kernel_profile,
  }));
  const locked = running || submitting;
  return (
    <>
      <PageHead
        title="性能与质量对照"
        kicker="FIXED INPUTS / MULTI-SCHEME COMPARISON"
        description="固定典型输入，串行执行多种推理方案，按同一用例核对输出质量、生成性能与实际生效。"
        extra={
          <Link to="/playground">
            <Button>打开交互推理案例</Button>
          </Link>
        }
      />
      {identity && (
        <div className="toolbar">
          <Button onClick={() => setConfigurationOpen((value) => !value)}>
            {configurationOpen ? '收起新对照配置' : '配置下一轮对照'}
          </Button>
          <span className="help-text">下方展示已冻结的对照计划与结果。</span>
        </div>
      )}
      {(!identity || configurationOpen) && (
        <Panel title="新对照配置" subtitle="已预选典型用例与方案，可直接开始；不会自动运行模型。">
          <QueryState loading={catalog.isPending} error={catalog.error ?? deployments.error} />
          <div className="benchmark-controls">
            <div>
              <label className="field-label" htmlFor="benchmark-deployment">
                模型部署
              </label>
              <Select
                id="benchmark-deployment"
                value={deployment?.id}
                onChange={(value) => {
                  setSelectedDeployment(value);
                  setCandidates(null);
                  setBaseline('framework-native');
                }}
                disabled={locked}
                options={deployments.data?.items
                  .filter((item) => item.provider === 'pytorch')
                  .map((item) => ({ value: item.id, label: item.name }))}
                className="wide-select"
              />
            </div>
            <div className="benchmark-baseline">
              <label className="field-label" htmlFor="benchmark-baseline">
                基线方案
              </label>
              <Select
                id="benchmark-baseline"
                value={baseline}
                onChange={setBaseline}
                disabled={locked}
                options={options}
                className="wide-select"
              />
            </div>
            <div className="benchmark-candidates">
              <label className="field-label" htmlFor="benchmark-candidates">
                候选方案（最多 3 个）
              </label>
              <Select
                id="benchmark-candidates"
                mode="multiple"
                maxCount={3}
                value={candidates}
                onChange={setCandidates}
                disabled={locked}
                options={options?.filter((item) => item.value !== baseline)}
                className="wide-select"
              />
            </div>
            <div>
              <label className="field-label" htmlFor="benchmark-budget">
                每次输出预算 / tokens
              </label>
              <InputNumber
                id="benchmark-budget"
                min={8}
                max={Math.min(128, deployment?.max_output_tokens ?? 128)}
                value={budget}
                onChange={(value) => setBudget(value ?? 64)}
                disabled={locked}
                className="wide-select"
              />
            </div>
            <div>
              <label className="field-label" htmlFor="benchmark-repeats">
                每个用例测量次数
              </label>
              <Select
                id="benchmark-repeats"
                value={repeats}
                onChange={setRepeats}
                disabled={locked}
                options={[1, 3, 5].map((value) => ({
                  value,
                  label: `${value} 次${value === 1 ? ' · 快速预览' : ''}`,
                }))}
                className="wide-select"
              />
            </div>
          </div>
          <Space wrap className="section-gap">
            <Checkbox
              checked={diagnostics}
              disabled={locked}
              onChange={(event) => setDiagnostics(event.target.checked)}
            >
              另做内存 / KV / 系统页诊断
            </Checkbox>
            <Tag>每用例预热 1 次，预热不计入统计</Tag>
            <Tag>相同输出预算 · greedy</Tag>
          </Space>
          <Table<FixedCase>
            size="small"
            rowKey="id"
            pagination={false}
            dataSource={catalog.data?.cases ?? []}
            rowSelection={{
              selectedRowKeys: cases,
              onChange: (keys) => setCases(keys.map(String)),
              getCheckboxProps: () => ({ disabled: locked }),
            }}
            expandable={{
              expandedRowRender: (item) => (
                <>
                  <pre className="benchmark-prompt">
                    {item.messages
                      .map((message) => `${message.role}: ${message.content}`)
                      .join('\n\n')}
                  </pre>
                  <div className="help-text mono">固定输入 SHA-256：{item.input_sha256}</div>
                </>
              ),
            }}
            columns={[
              { title: '固定输入', dataIndex: 'name' },
              { title: '负载类型', dataIndex: 'category' },
              {
                title: '质量评分',
                render: (_, item) =>
                  item.scoring === 'unscored'
                    ? '开放任务 · 查看输出，不自动判优'
                    : item.scoring === 'json'
                      ? 'JSON 语义与字段类型'
                      : '固定参考答案 exact-match',
              },
              {
                title: '参考答案',
                render: (_, item) =>
                  item.expected == null ? '无自动参考分' : JSON.stringify(item.expected),
              },
            ]}
            scroll={{ x: 650 }}
          />
          <div className="benchmark-launch section-gap">
            <div className="help-text">
              选择 {1 + candidates.length} 个方案 × {cases.length} 个用例；每用例 {repeats}{' '}
              次测量。切换算子时自动重载，结束或取消后恢复原部署。测试运行在服务端，刷新或离开页面不会重提或停止任务。
            </div>
            <Button
              type="primary"
              loading={submitting}
              disabled={locked || !catalog.data || !cases.length || !candidates.length}
              onClick={() => void launch()}
            >
              开始固定输入对照
            </Button>
          </div>
          {catalog.data?.active_id && catalog.data.active_id !== identity && (
            <Alert
              type="info"
              message="已有对照任务正在执行"
              action={
                <Button onClick={() => setParams({ benchmark: catalog.data!.active_id! })}>
                  查看正在运行的对照
                </Button>
              }
            />
          )}
          {error && <Alert type="error" showIcon message={error} />}
        </Panel>
      )}
      <Panel
        title="对照结果"
        subtitle="历史记录与冻结计划绑定，修改上方配置不改变已有结果。"
        extra={
          <Select
            aria-label="选择历史性能对照"
            placeholder="选择历史对照"
            value={identity || undefined}
            onChange={(value) => setParams({ benchmark: value })}
            style={{ width: 300, maxWidth: '100%' }}
            options={listing.data?.items.map((item) => ({
              value: item.id,
              label: `${date(item.created_at)} · ${item.state}`,
            }))}
          />
        }
      >
        <QueryState
          loading={result.isFetching && !result.data}
          error={result.error ?? listing.error}
        />
        {!identity && (
          <div className="benchmark-empty">
            <h3>运行一次固定输入对照，建立可重复的基线</h3>
            <p>
              默认展示真实固定用例，尚无测量数据。执行后按用例比较多个方案，输出、正确率与耗时使用不同列。
            </p>
          </div>
        )}
        {result.data && (
          <>
            <div className="toolbar">
              <Space wrap>
                <Status state={result.data.state} />
                <strong>{phaseLabel[result.data.phase] ?? result.data.phase}</strong>
                <code>{result.data.id}</code>
                <span>
                  完成测量{' '}
                  {
                    result.data.records.filter(
                      (row) =>
                        row.role === 'measurement' &&
                        row.state === 'completed' &&
                        row.cleanup === 'succeeded',
                    ).length
                  }{' '}
                  /{' '}
                  {result.data.plan.schemes.length *
                    result.data.plan.cases.length *
                    result.data.plan.repeats}
                </span>
              </Space>
              {!finished(result.data.state) && (
                <Button
                  danger
                  disabled={result.data.state === 'cancelling'}
                  onClick={() => void cancel()}
                >
                  取消对照并恢复部署
                </Button>
              )}
            </div>
            {result.data.current_run_id && (
              <Link to={`/runs/${result.data.current_run_id}`}>查看当前执行任务</Link>
            )}
            {result.data.error && <Alert type="error" message={result.data.error} />}
            {result.data.restore_error && (
              <Alert
                type="error"
                message="原部署未能恢复，请检查设备与部署"
                description={result.data.restore_error}
              />
            )}
            <Results key={result.data.id} benchmark={result.data} />
          </>
        )}
      </Panel>
    </>
  );
}
