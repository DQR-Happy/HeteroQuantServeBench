import { useCallback, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Alert, Select, Space, Table, Tag } from 'antd';
import Button from '../components/ActionButton';
import { api, base } from '../api/client';
import DMATChart from './DMATChart';

type Cell = {
  row: number;
  column: number;
  object?: number;
  level?: number;
  events: number;
  reads: number;
  writes: number;
  instruction_bytes: number;
};
export type DMATObject = {
  id: number;
  name: string;
  role: string;
  shape?: number[];
  color: string;
  events: number;
  pages: number;
  instruction_bytes: number;
  color_thresholds: number[];
};
export type DMATLaunch = {
  kernel: string;
  trace: string;
  svg: string;
  grid: number[];
  block: number[];
  context?: { layer?: string; phase?: string; block_n?: number };
  probe?: { event_bytes?: number };
  plot: {
    events: number;
    visible_events?: number;
    attempted_events: number;
    dropped_events: number;
    overflow_threads: number;
    complete: boolean;
    duration_ns: number;
    duration?: number;
    time_bin_ns: number;
    time_bin?: number;
    time_bins: number;
    time_start?: number;
    time_end?: number;
    clock_domain?: string;
    time_unit?: string;
    address_bin_bytes: number;
    requested_address_bin_bytes: number;
    addresses: string[];
    address_axis?: string;
    address_rows?: { label: string; address: string; object: number }[];
    objects?: DMATObject[];
    unknown_events?: number;
    cells: Cell[];
    sites: { site: number; opcode: string; operation: string; width: number; events: number }[];
  };
};
export type DMATData = { mode?: string; memory_launches?: DMATLaunch[] };

export default function DMATPanel({ data, run }: { data: DMATData; run: string }) {
  const [index, setIndex] = useState(0);
  const launches = data.memory_launches ?? [];
  const original = launches[Math.min(index, launches.length - 1)];
  const [metric, setMetric] = useState<'events' | 'reads' | 'writes'>('events');
  const [clock, setClock] = useState(original?.plot.clock_domain ?? 'GPU_globaltimer');
  const [pages, setPages] = useState(original?.plot.address_bin_bytes ?? 65536);
  const [window, setWindow] = useState<[number, number]>();
  const [viewEnabled, setViewEnabled] = useState(original?.probe?.event_bytes === 48);
  const onWindow = useCallback((start: number, end: number) => {
    setWindow([start, end]);
    setViewEnabled(true);
  }, []);
  const onFullWindow = useCallback(() => {
    setWindow(undefined);
    setViewEnabled(true);
  }, []);
  const view = useQuery({
    queryKey: ['dmat-view', run, index, clock, pages, window],
    enabled: viewEnabled,
    queryFn: ({ signal }) =>
      api<DMATLaunch>(
        `/diagnostics/jobs/${run}/memory/${index}?clock=${clock}&page_bytes=${pages}&time_bins=160${window ? `&start=${window[0]}&end=${window[1]}` : ''}`,
        undefined,
        signal,
      ),
    retry: false,
    placeholderData: (previous, query) =>
      query?.queryKey[1] === run &&
      query.queryKey[2] === index &&
      query.queryKey[3] === clock &&
      query.queryKey[4] === pages
        ? previous
        : undefined,
  });
  const launch = viewEnabled && view.data ? view.data : original;
  if (!launch) return <p>当前采集等待实际访存事件，请查看采集覆盖和探针日志。</p>;
  const plot = launch.plot;
  const artifact = (path: string) => `${base}/diagnostics/jobs/${run}/artifacts/${path}`;
  return (
    <section aria-label="真实访存 DMAT">
      <h3>真实访存密度图 · DMAT</h3>
      <Space wrap>
        <Select
          aria-label="DMAT Kernel 启动"
          value={Math.min(index, launches.length - 1)}
          onChange={(v) => {
            setIndex(v);
            setWindow(undefined);
            setViewEnabled(true);
          }}
          style={{ minWidth: 340, maxWidth: '100%' }}
          options={launches.map((x, i) => ({
            value: i,
            label: `${i + 1} · ${x.context?.layer ?? x.kernel} · ${x.context?.phase ?? ''}`,
          }))}
        />
        <Select
          aria-label="DMAT 时间口径"
          value={clock}
          onChange={(v) => {
            setClock(v);
            setWindow(undefined);
            setViewEnabled(true);
          }}
          options={[
            { value: 'thread_entry_clock64', label: '线程入口归一化 · cycles（论文图 1 / 11）' },
            { value: 'GPU_globaltimer', label: 'GPU 全局时间 · ns（论文图 15）' },
          ]}
        />
        <Select
          aria-label="DMAT 地址粒度"
          value={pages}
          onChange={(v) => {
            setPages(v);
            setViewEnabled(true);
          }}
          options={[4096, 16384, 65536].map((v) => ({ value: v, label: `${v / 1024} KiB` }))}
        />
        <Select
          aria-label="DMAT 访问类型"
          value={metric}
          onChange={setMetric}
          options={[
            { value: 'events', label: '全部访存' },
            { value: 'reads', label: '读取' },
            { value: 'writes', label: '写入' },
          ]}
        />
        <Button onClick={onFullWindow}>完整时间范围</Button>
        <Tag>{plot.events.toLocaleString()} 条实际事件</Tag>
        <Tag>
          {plot.complete ? '容量内完整采集' : `丢失 ${plot.dropped_events.toLocaleString()} 条事件`}
        </Tag>
      </Space>
      {view.isFetching && <p>正在从原始访存事件重新聚合…</p>}
      {view.error && <Alert type="error" message={(view.error as Error).message} />}
      {!plot.complete && (
        <Alert
          type="warning"
          message={`${plot.overflow_threads} 个线程超出容量；增大每线程容量后重新采集。`}
        />
      )}
      <p>
        横轴：
        {plot.clock_domain === 'thread_entry_clock64'
          ? '每个线程相对入口的时钟周期'
          : 'GPU 全局计时器，实际分辨率由设备决定'}
        。纵轴按张量及已观测虚拟页组织，地址空隙折叠。颜色区分张量，深浅采用该张量非空单元密度的五档分位数；悬停查看原始计数。
      </p>
      <p>
        Grid {launch.grid.join('×')} · Block {launch.block.join('×')} · 时间粒度{' '}
        {plot.time_bin ?? plot.time_bin_ns} {plot.time_unit ?? 'ns'} · 地址粒度{' '}
        {plot.address_bin_bytes}{' '}
        B。点击“聚焦查看”可独立缩放和平移；选中时间范围后可读取原始事件重新分箱。
      </p>
      <Space wrap>
        {plot.objects?.map((o) => (
          <Tag key={o.id} color={o.color}>
            {o.role} {o.shape?.join('×')} · {o.events.toLocaleString()} 次
          </Tag>
        ))}
      </Space>
      <DMATChart
        launch={launch}
        metric={metric}
        onWindow={onWindow}
        onFullWindow={onFullWindow}
        busy={view.isFetching}
        error={view.error?.message}
      />
      {!!plot.unknown_events && (
        <Alert
          type="warning"
          message={`${plot.unknown_events} 条事件的张量归属等待适配；Unknown 区域保留真实地址。`}
        />
      )}
      <p>
        密度统计启用线程执行的全局访存指令。页为软件地址分箱；访存字节按指令宽度累计。采集会扰动执行时间，性能结论请查看独立的无探针评测。
      </p>
      <Space wrap>
        <a href={artifact(launch.svg)} download>
          下载全部访存 SVG
        </a>
        <a href={artifact(launch.trace)} download>
          下载原始访存事件
        </a>
      </Space>
      <Table
        size="small"
        rowKey="site"
        dataSource={plot.sites}
        pagination={{ pageSize: 8 }}
        columns={[
          { title: '指令位置', dataIndex: 'site' },
          { title: 'PTX 访存指令', dataIndex: 'opcode' },
          { title: '读写', dataIndex: 'operation' },
          { title: '每事件字节', dataIndex: 'width' },
          { title: '实际事件', dataIndex: 'events' },
        ]}
      />
    </section>
  );
}
