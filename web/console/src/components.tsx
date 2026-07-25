import { useEffect, useRef, type ReactNode } from 'react';
import { Alert, Empty, Spin, Tag } from 'antd';
import * as echarts from 'echarts/core';
import { BarChart, LineChart } from 'echarts/charts';
import {
  AriaComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { EChartsOption } from 'echarts';
import { fmt } from './api/client';

echarts.use([
  BarChart,
  LineChart,
  AriaComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  CanvasRenderer,
]);

const labels: Record<string, string> = {
  ready: '已就绪',
  unloaded: '未加载',
  loading: '加载中',
  draining: '排空中',
  queued: '排队中',
  running: '执行中',
  completed: '已完成',
  failed: '失败',
  cancelled: '已取消',
  cancel_requested: '取消中',
  timed_out: '超时',
  interrupted: '执行中断',
  PASS: '通过',
  FAIL: '未通过',
  BLOCKED: '受阻',
  UNKNOWN: '未标注',
  NOT_EXECUTED: '未执行',
};
export function Status({ state }: { state: string }) {
  const color = ['ready', 'completed', 'PASS'].includes(state)
    ? 'success'
    : ['failed', 'FAIL', 'timed_out'].includes(state)
      ? 'error'
      : ['loading', 'running', 'queued', 'cancel_requested'].includes(state)
        ? 'processing'
        : 'default';
  return <Tag color={color}>{labels[state] ?? state}</Tag>;
}
export function PageHead({
  kicker,
  title,
  description,
  extra,
}: {
  kicker: string;
  title: string;
  description: string;
  extra?: ReactNode;
}) {
  return (
    <div className="page-head">
      <div>
        <div className="eyebrow">{kicker}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      <div>{extra}</div>
    </div>
  );
}
export function Panel({
  title,
  subtitle,
  children,
  extra,
  className = '',
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  extra?: ReactNode;
  className?: string;
}) {
  return (
    <section className={'panel ' + className}>
      {title && (
        <div className="panel-head">
          <div>
            <h2>{title}</h2>
            {subtitle && <p>{subtitle}</p>}
          </div>
          {extra}
        </div>
      )}
      {children}
    </section>
  );
}
export function Stat({
  label,
  value,
  unit = '',
  hint,
  accent,
}: {
  label: string;
  value: number | string | null | undefined;
  unit?: string;
  hint?: string;
  accent?: boolean;
}) {
  return (
    <div className={'stat ' + (accent ? 'accent' : '')}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">
        {typeof value === 'number' ? fmt(value) : (value ?? '—')}
        <small>{unit}</small>
      </div>
      <div className="stat-hint">{hint ?? '未采集的指标显示为 —'}</div>
    </div>
  );
}
export function Chart({
  option,
  height = 250,
  label,
}: {
  option: EChartsOption;
  height?: number;
  label: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current, undefined, { renderer: 'canvas' });
    chart.setOption({ ...option, aria: { enabled: true, description: label } });
    const resize = new ResizeObserver(() => chart.resize());
    resize.observe(ref.current);
    return () => {
      resize.disconnect();
      chart.dispose();
    };
  }, [option, label]);
  return <div ref={ref} role="img" aria-label={label} style={{ height, width: '100%' }} />;
}
export function QueryState({
  loading,
  error,
  empty,
}: {
  loading?: boolean;
  error?: unknown;
  empty?: boolean;
}) {
  if (loading)
    return (
      <div className="loading">
        <Spin />
        <span>正在读取真实数据…</span>
      </div>
    );
  if (error)
    return (
      <Alert
        showIcon
        type="error"
        message="数据读取失败"
        description={error instanceof Error ? error.message : String(error)}
      />
    );
  if (empty)
    return <Empty description="尚无运行记录。加载部署后，在推理工作台开始第一次真实运行。" />;
  return null;
}
export function JsonView({ value }: { value: unknown }) {
  const text = JSON.stringify(value, null, 2);
  return (
    <pre className="json-view">
      {text?.length > 60000 ? text.slice(0, 60000) + '\n… 预览已截断，请下载完整证据。' : text}
    </pre>
  );
}
