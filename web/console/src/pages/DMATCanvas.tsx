import { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import type { DMATLaunch } from './DMATPanel';
import { percentRange, panRange, type DMATViewport } from './dmatViewport';

const grid = { left: 185, right: 55, top: 36, bottom: 86 };

export default function DMATCanvas({
  launch,
  metric = 'events',
  viewport,
  onViewport,
  expanded = false,
}: {
  launch: DMATLaunch;
  metric?: 'events' | 'reads' | 'writes';
  viewport?: DMATViewport;
  onViewport?: (value: DMATViewport) => void;
  expanded?: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const instance = useRef<echarts.ECharts | null>(null);
  const currentViewport = useRef(viewport);
  const changeViewport = useRef(onViewport);
  currentViewport.current = viewport;
  changeViewport.current = onViewport;
  const drag = useRef<{ x: number; y: number; viewport: DMATViewport } | undefined>(undefined);
  const plot = launch.plot;
  useEffect(() => {
    if (!host.current) return;
    host.current.dataset.rendered = 'false';
    const chart = echarts.init(host.current);
    instance.current = chart;
    chart.on('finished', () => {
      if (host.current) host.current.dataset.rendered = 'true';
    });
    const cells = plot.cells.filter((c) => c[metric] > 0);
    const objects = plot.objects ?? [];
    const max = cells.reduce((value, cell) => Math.max(value, cell[metric]), 1);
    const quantum = plot.time_bin ?? plot.time_bin_ns;
    const start = plot.time_start ?? 0;
    const unit = plot.time_unit ?? 'ns';
    chart.setOption({
      animation: false,
      aria: { enabled: true, description: '真实访存密度热力图' },
      grid: expanded ? grid : { left: 185, right: 25, top: 25, bottom: 65 },
      tooltip: {
        confine: true,
        formatter: (p: unknown) => {
          const c = cells[(p as { dataIndex: number }).dataIndex];
          return c
            ? `${echarts.format.encodeHTML(plot.address_rows?.[c.row]?.label ?? plot.addresses[c.row])}<br/>VA ${echarts.format.encodeHTML(plot.addresses[c.row])}<br/>${start + c.column * quantum}–${start + (c.column + 1) * quantum} ${unit}<br/>事件 ${c.events} · 读 ${c.reads} · 写 ${c.writes}<br/>指令访问字节 ${c.instruction_bytes}`
            : '';
        },
      },
      xAxis: {
        type: 'category',
        name:
          plot.clock_domain === 'thread_entry_clock64'
            ? '相对线程入口 / cycles'
            : '相对首个访存事件 / ns',
        nameLocation: 'middle',
        nameGap: 32,
        data: Array.from({ length: plot.time_bins }, (_, i) => start + i * quantum),
        axisLabel: { hideOverlap: true },
      },
      yAxis: {
        type: 'category',
        name: `${plot.address_axis === 'tensor_relative_byte_bins' ? '张量相对地址' : '张量 / GPU VA'} · ${plot.address_bin_bytes} B`,
        inverse: true,
        data: plot.address_rows?.map((r) => r.label) ?? plot.addresses,
        axisLabel: { fontFamily: 'monospace', hideOverlap: true, fontSize: 10 },
      },
      dataZoom: expanded
        ? [
            {
              id: 'dmat-time',
              type: 'slider',
              xAxisIndex: 0,
              bottom: 5,
              height: 20,
              filterMode: 'none',
              showDetail: false,
              brushSelect: false,
            },
            {
              id: 'dmat-address',
              type: 'slider',
              yAxisIndex: 0,
              right: 8,
              width: 16,
              top: grid.top,
              bottom: grid.bottom,
              filterMode: 'none',
              showDetail: false,
              brushSelect: false,
            },
          ]
        : [],
      series: [
        {
          type: 'heatmap',
          progressive: 2000,
          data: cells.map((c) => {
            const obj = objects.find((o) => o.id === c.object);
            return {
              value: [c.column, c.row, c[metric]],
              itemStyle: {
                color: obj?.color ?? '#326ea5',
                opacity: c.level ? c.level / 5 : 0.15 + (0.85 * c[metric]) / max,
              },
            };
          }),
          emphasis: { itemStyle: { borderColor: '#263d50', borderWidth: 1 } },
        },
      ],
    });
    chart.on('datazoom', () => {
      const ranges = chart.getOption().dataZoom as { id: string; start: number; end: number }[];
      const time = ranges.find((r) => r.id === 'dmat-time');
      const address = ranges.find((r) => r.id === 'dmat-address');
      if (time && address)
        changeViewport.current?.({
          time: percentRange(time.start, time.end, plot.time_bins),
          address: percentRange(address.start, address.end, plot.addresses.length),
        });
    });
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(host.current);
    return () => {
      observer.disconnect();
      chart.dispose();
      instance.current = null;
    };
  }, [plot, metric, expanded]);
  useEffect(() => {
    if (!viewport || !instance.current) return;
    const percentage = (value: number, count: number) =>
      count > 1 ? (value * 100) / (count - 1) : 0;
    instance.current.setOption(
      {
        dataZoom: [
          {
            id: 'dmat-time',
            start: percentage(viewport.time[0], plot.time_bins),
            end: plot.time_bins > 1 ? percentage(viewport.time[1], plot.time_bins) : 100,
          },
          {
            id: 'dmat-address',
            start: percentage(viewport.address[0], plot.addresses.length),
            end:
              plot.addresses.length > 1
                ? percentage(viewport.address[1], plot.addresses.length)
                : 100,
          },
        ],
      },
      { silent: true },
    );
  }, [viewport, plot, metric]);
  return (
    <div
      ref={host}
      role="img"
      aria-label="真实访存密度热力图"
      data-time-start={viewport?.time[0]}
      data-time-end={viewport?.time[1]}
      data-address-start={viewport?.address[0]}
      data-address-end={viewport?.address[1]}
      onPointerDown={(event) => {
        if (!currentViewport.current || event.button !== 0) return;
        const bounds = event.currentTarget.getBoundingClientRect();
        const x = event.clientX - bounds.left;
        const y = event.clientY - bounds.top;
        if (
          x < grid.left ||
          x > bounds.width - grid.right ||
          y < grid.top ||
          y > bounds.height - grid.bottom
        )
          return;
        drag.current = { x: event.clientX, y: event.clientY, viewport: currentViewport.current };
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => {
        const from = drag.current;
        if (!from) return;
        const bounds = event.currentTarget.getBoundingClientRect();
        changeViewport.current?.({
          time: panRange(
            from.viewport.time,
            ((from.x - event.clientX) * (from.viewport.time[1] - from.viewport.time[0] + 1)) /
              Math.max(1, bounds.width - grid.left - grid.right),
            plot.time_bins,
          ),
          address: panRange(
            from.viewport.address,
            ((from.y - event.clientY) * (from.viewport.address[1] - from.viewport.address[0] + 1)) /
              Math.max(1, bounds.height - grid.top - grid.bottom),
            plot.addresses.length,
          ),
        });
      }}
      onPointerUp={() => {
        drag.current = undefined;
      }}
      onPointerCancel={() => {
        drag.current = undefined;
      }}
      onLostPointerCapture={() => {
        drag.current = undefined;
      }}
      style={{
        height: expanded ? '100%' : Math.min(700, Math.max(260, 125 + plot.addresses.length * 5)),
        width: '100%',
        minWidth: 280,
        minHeight: expanded ? 260 : undefined,
        cursor: expanded ? 'grab' : undefined,
        touchAction: expanded ? 'none' : undefined,
      }}
    />
  );
}
