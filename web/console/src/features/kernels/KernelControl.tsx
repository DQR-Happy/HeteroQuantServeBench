import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Button, Select } from 'antd';
import { Link } from 'react-router-dom';
import { api } from '../../api/client';
import type { Deployment, Run } from '../../api/types';
import { JsonView, QueryState, Status } from '../../components';
import { applyKernelProfile } from './lifecycle';
import type { KernelCatalogue } from './types';

const unavailable = (reason?: string | null) =>
  reason === 'KERNEL_NATIVE_MANIFEST_REQUIRED'
    ? '服务尚未配置原生算子制品'
    : reason === 'EXTERNAL_PROVIDER_MANAGES_KERNELS'
      ? '算子由外部引擎管理'
      : reason === 'KERNEL_MODEL_SHAPE_UNSUPPORTED'
        ? '此实现仅适配 Qwen3 1.7B，当前模型使用框架原生实现'
        : reason === 'KERNEL_MODEL_CONFIG_INVALID'
          ? '模型配置读取失败，无法确认算子兼容性'
      : '原生制品身份检查未通过，请检查服务配置';

export default function KernelControl({
  deployment,
  selected,
  onSelect,
  disabled,
  onBusyChange,
}: {
  deployment: Deployment;
  selected: string;
  onSelect: (value: string) => void;
  disabled: boolean;
  onBusyChange: (value: boolean) => void;
}) {
  const client = useQueryClient();
  const controller = useRef<AbortController | null>(null);
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState<Run>();
  const [error, setError] = useState('');
  const [applied, setApplied] = useState('');
  const query = useQuery({
    queryKey: ['kernel-profiles', deployment.id],
    queryFn: () =>
      api<KernelCatalogue>(`/kernels/profiles?deployment_id=${encodeURIComponent(deployment.id)}`),
    staleTime: 30000,
  });
  useEffect(() => () => controller.current?.abort(), []);
  const profiles = query.data?.profiles ?? [];
  const chosen = profiles.find((item) => item.profile_id === selected);
  const current = deployment.kernel_profile ?? 'framework-native';
  const dirty = selected !== current;
  const canApply = chosen?.availability?.loadable === true;
  const groups = Array.from(new Set(profiles.map((item) => item.family)));
  const apply = async () => {
    if (controller.current || disabled || !canApply) return;
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    onBusyChange(true);
    setError('');
    setApplied('');
    try {
      const actual = await applyKernelProfile(deployment, selected, abort.signal, setJob);
      if (abort.signal.aborted) return;
      client.setQueryData<{ items: Deployment[] }>(['deployments'], (old) =>
        old ? { items: old.items.map((item) => (item.id === actual.id ? actual : item)) } : old,
      );
      setApplied(actual.kernel_profile ?? 'framework-native');
    } catch (failure) {
      if (!abort.signal.aborted) setError((failure as Error).message);
    } finally {
      if (!abort.signal.aborted) {
        controller.current = null;
        setBusy(false);
        onBusyChange(false);
        void client.invalidateQueries({ queryKey: ['deployments'] });
      }
    }
  };
  return (
    <section className="kernel-control" aria-label="算子实现与模型生命周期">
      <label className="field-label" htmlFor="kernel-profile">
        算子实现与版本
      </label>
      <QueryState loading={query.isPending} error={query.error} />
      {query.isError && (
        <Button size="small" onClick={() => void query.refetch()}>
          重试读取算子目录
        </Button>
      )}
      <Select
        id="kernel-profile"
        aria-label="算子配置"
        value={selected}
        onChange={(value) => {
          onSelect(value);
          setApplied('');
          setError('');
        }}
        disabled={disabled || busy || query.isPending || query.isError}
        style={{ width: '100%' }}
        showSearch
        optionFilterProp="label"
        options={groups.map((family) => ({
          label: family,
          options: profiles
            .filter((item) => item.family === family)
            .map((item) => ({
              value: item.profile_id,
              label: item.label,
              disabled: item.availability?.loadable !== true,
              title: item.availability?.loadable
                ? item.profile_id
                : unavailable(item.availability?.reason),
            })),
        }))}
      />
      {chosen && (
        <div className="kernel-selection">
          <code>{chosen.profile_id}</code>
          <div className="help-text">
            {Object.entries(chosen.operators).length
              ? Object.entries(chosen.operators).map(([name, variant]) => (
                  <div key={name}>
                    {name} → <code>{variant}</code>
                  </div>
                ))
              : '全部使用模型的框架原生实现。'}
          </div>
          <p className="help-text">未列出的算子保留原实现。列入目录不代表质量或性能验收通过。</p>
        </div>
      )}
      {query.data?.native_artifact && (
        <div className="help-text">
          原生制品：
          {query.data.native_artifact.status === 'files_verified'
            ? '文件身份已核验，运行兼容性在加载时验证'
            : unavailable(query.data.native_artifact.reason)}
        </div>
      )}
      <div className="kernel-current">
        <strong>当前已加载方案</strong>
        <code>{deployment.state === 'ready' ? current : '模型尚未就绪'}</code>
        {deployment.epoch && <span className="muted small mono">epoch {deployment.epoch}</span>}
      </div>
      {dirty && deployment.state === 'ready' && (
        <Alert type="info" showIcon message="选择尚未应用，需要重载模型" />
      )}
      <Button
        block
        loading={busy}
        disabled={
          disabled ||
          busy ||
          !canApply ||
          !['ready', 'unloaded', 'failed'].includes(deployment.state)
        }
        onClick={() => void apply()}
      >
        {busy
          ? job?.kind === 'unload'
            ? '正在排空并卸载模型'
            : '正在加载与预热模型'
          : deployment.state === 'ready'
            ? dirty
              ? '应用并重载模型'
              : '重新加载当前方案'
            : '切换模型并加载方案'}
      </Button>
      <div className="help-text">
        切换会排空并卸载当前 GPU 模型，再加载所选模型与方案。加载成功后，仍需单独提交推理或固定输入评测。离开此页会停止后续自动提交；已接受任务可在运行记录查看。
      </div>
      {(job || deployment.load_run_id) && (
        <div className="help-text">
          {job && <Status state={job.state} />}{' '}
          <Link to={`/runs/${job?.id ?? deployment.load_run_id}`}>查看模型生命周期任务</Link>
        </div>
      )}
      {error && <Alert type="error" showIcon message="算子方案未应用成功" description={error} />}
      {deployment.last_error && !error && (
        <Alert type="error" showIcon message={deployment.last_error} />
      )}
      {applied && deployment.state === 'ready' && current === applied && (
        <Alert type="success" showIcon message={`模型已就绪：${applied}`} />
      )}
      {deployment.detail.kernel_plan && (
        <details>
          <summary>已加载计划与制品身份</summary>
          <JsonView value={deployment.detail.kernel_plan} />
        </details>
      )}
    </section>
  );
}
