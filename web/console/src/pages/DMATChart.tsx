import { useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Modal, Select, Space, Tag } from 'antd';
import { ExpandOutlined } from '@ant-design/icons';
import Button from '../components/ActionButton';
import DMATCanvas from './DMATCanvas';
import type { DMATLaunch } from './DMATPanel';
import { fullRange, panRange, timeWindow, zoomRange, type DMATViewport } from './dmatViewport';
import './DMATChart.css';

type Props = {
  launch: DMATLaunch;
  metric?: 'events' | 'reads' | 'writes';
  title?: string;
  onWindow?: (start: number, end: number) => void;
  onFullWindow?: () => void;
  busy?: boolean;
  error?: string;
};
type Axis = keyof DMATViewport;
type WheelMode = 'address-pan' | 'time-pan' | 'time-zoom' | 'address-zoom';

function FocusedView({ launch, metric, onWindow, onFullWindow, busy, error }: Props) {
  const plot = launch.plot;
  const full = useMemo(
    () => ({ time: fullRange(plot.time_bins), address: fullRange(plot.addresses.length) }),
    [plot],
  );
  const [selection, setSelection] = useState({ plot, viewport: full });
  // A newly aggregated grid owns its own indices. Polling the parent preserves
  // the existing viewport while a successful rebin starts at the new full grid.
  const viewport = selection.plot === plot ? selection.viewport : full;
  if (selection.plot !== plot) setSelection({ plot, viewport: full });
  const [wheel, setWheel] = useState<WheelMode>('address-pan');
  const area = useRef<HTMLDivElement>(null);
  const update = (value: DMATViewport) => setSelection({ plot, viewport: value });
  const transform = (axis: Axis, mode: 'pan' | 'zoom', amount: number) => {
    setSelection((previous) => {
      const current = previous.plot === plot ? previous.viewport : full;
      const count = axis === 'time' ? plot.time_bins : plot.addresses.length;
      const range = current[axis];
      const next =
        mode === 'zoom'
          ? zoomRange(range, amount, count)
          : panRange(range, amount * Math.max(1, Math.round((range[1] - range[0] + 1) / 4)), count);
      return { plot, viewport: { ...current, [axis]: next } };
    });
  };
  const wheelAction = useRef({ transform, wheel });
  wheelAction.current = { transform, wheel };
  useEffect(() => {
    const node = area.current;
    if (!node) return;
    let accumulated = 0;
    let lastMode = '';
    let lastTime = 0;
    const onWheel = (event: WheelEvent) => {
      if (event.ctrlKey || event.metaKey) return;
      event.preventDefault();
      event.stopPropagation();
      const delta =
        (event.deltaY || event.deltaX) *
        (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? node.clientHeight : 1);
      if (!delta) return;
      const mode = event.shiftKey ? 'time-pan' : wheelAction.current.wheel;
      // Accumulate small trackpad deltas so a gesture still moves at high zoom.
      if (mode !== lastMode || event.timeStamp - lastTime > 200 || accumulated * delta < 0)
        accumulated = 0;
      lastMode = mode;
      lastTime = event.timeStamp;
      accumulated += delta;
      const steps = Math.trunc(accumulated / 40);
      if (!steps) return;
      accumulated -= steps * 40;
      const axis = mode.startsWith('time') ? 'time' : 'address';
      const zoom = mode.endsWith('zoom');
      wheelAction.current.transform(
        axis,
        zoom ? 'zoom' : 'pan',
        zoom
          ? Math.pow(1.12, Math.max(-5, Math.min(5, steps)))
          : Math.sign(steps) * Math.min(1, Math.abs(steps) / 2),
      );
    };
    node.addEventListener('wheel', onWheel, { passive: false });
    return () => node.removeEventListener('wheel', onWheel);
  }, []);
  const quantum = plot.time_bin ?? plot.time_bin_ns;
  const start = plot.time_start ?? 0;
  const range = timeWindow(
    viewport.time,
    start,
    quantum,
    plot.time_end ?? start + plot.time_bins * quantum,
  );
  const unit = plot.time_unit ?? 'ns';
  const narrowed = viewport.time[0] > 0 || viewport.time[1] < plot.time_bins - 1;
  const addressCount = plot.addresses.length ? viewport.address[1] - viewport.address[0] + 1 : 0;
  return (
    <>
      <div className="dmat-focus-toolbar">
        <Space wrap size={8}>
          <span className="dmat-control-label">时间</span>
          <Button size="small" aria-label="时间缩小" onClick={() => transform('time', 'zoom', 2)}>
            −
          </Button>
          <Button size="small" aria-label="时间放大" onClick={() => transform('time', 'zoom', 0.5)}>
            ＋
          </Button>
          <Button size="small" aria-label="时间左移" onClick={() => transform('time', 'pan', -1)}>
            ←
          </Button>
          <Button size="small" aria-label="时间右移" onClick={() => transform('time', 'pan', 1)}>
            →
          </Button>
        </Space>
        <Space wrap size={8}>
          <span className="dmat-control-label">地址</span>
          <Button
            size="small"
            aria-label="地址缩小"
            onClick={() => transform('address', 'zoom', 2)}
          >
            −
          </Button>
          <Button
            size="small"
            aria-label="地址放大"
            onClick={() => transform('address', 'zoom', 0.5)}
          >
            ＋
          </Button>
          <Button
            size="small"
            aria-label="地址上移"
            onClick={() => transform('address', 'pan', -1)}
          >
            ↑
          </Button>
          <Button size="small" aria-label="地址下移" onClick={() => transform('address', 'pan', 1)}>
            ↓
          </Button>
        </Space>
        <Space wrap size={8}>
          <label htmlFor="dmat-wheel-mode">滚轮</label>
          <Select
            id="dmat-wheel-mode"
            aria-label="DMAT 滚轮操作"
            size="small"
            value={wheel}
            onChange={setWheel}
            style={{ width: 120 }}
            options={[
              { value: 'address-pan', label: '上下移地址' },
              { value: 'address-zoom', label: '缩放地址' },
              { value: 'time-zoom', label: '缩放时间' },
              { value: 'time-pan', label: '左右移时间' },
            ]}
          />
          <Button size="small" onClick={() => update(full)}>
            适应当前数据
          </Button>
        </Space>
      </div>
      <Space wrap size={[8, 4]}>
        {plot.objects?.map((o) => (
          <Tag key={o.id} color={o.color}>
            {o.role}
          </Tag>
        ))}
        <span className="dmat-focus-summary" role="status" aria-label="DMAT 当前视窗">
          时间 {range[0].toLocaleString()}–{range[1].toLocaleString()} {unit} · 地址行{' '}
          {addressCount ? viewport.address[0] + 1 : 0}–{addressCount ? viewport.address[1] + 1 : 0}{' '}
          / {plot.addresses.length} · {addressCount} 行
        </span>
      </Space>
      <div
        ref={area}
        className="dmat-focus-canvas"
        tabIndex={0}
        role="group"
        aria-label="DMAT 图表操作区"
        aria-describedby="dmat-focus-help"
        onKeyDown={(event) => {
          if (
            event.target !== event.currentTarget ||
            event.ctrlKey ||
            event.metaKey ||
            event.altKey
          )
            return;
          const actions: Record<string, () => void> = {
            ArrowUp: () => transform('address', 'pan', -1),
            ArrowDown: () => transform('address', 'pan', 1),
            ArrowLeft: () => transform('time', 'pan', -1),
            ArrowRight: () => transform('time', 'pan', 1),
            '+': () => transform('time', 'zoom', 0.5),
            '=': () => transform('time', 'zoom', 0.5),
            '-': () => transform('time', 'zoom', 2),
            Home: () => update(full),
          };
          if (actions[event.key]) {
            event.preventDefault();
            actions[event.key]();
          }
        }}
      >
        <DMATCanvas
          launch={launch}
          metric={metric}
          expanded
          viewport={viewport}
          onViewport={update}
        />
      </div>
      <p id="dmat-focus-help" className="dmat-focus-help">
        拖动画面平移；底部与右侧滑块调整范围。点击图表后，方向键平移，＋ / − 缩放时间，Home
        适应当前数据。Shift＋滚轮左右平移，Esc 返回页面。
      </p>
      <div className="dmat-focus-actions">
        <Space wrap>
          {onWindow && (
            <Button
              type="primary"
              size="small"
              disabled={busy || !narrowed || range[1] <= range[0]}
              disabledReason={{
                reason: busy
                  ? '正在读取原始事件，请等待当前重新分箱完成。'
                  : '先使用时间放大按钮或底部滑块选中较小的时间范围，再重新分箱。',
              }}
              onClick={() => onWindow(range[0], range[1])}
            >
              按此时间范围重新分箱
            </Button>
          )}
          {onFullWindow && (
            <Button
              size="small"
              disabled={busy}
              disabledReason={{ reason: '正在读取原始事件，请等待当前重新分箱完成。' }}
              onClick={onFullWindow}
            >
              完整时间范围
            </Button>
          )}
          <span className="dmat-focus-help">
            时间分箱 {quantum.toLocaleString()} {unit} · 地址分箱{' '}
            {plot.address_bin_bytes.toLocaleString()} B
          </span>
        </Space>
        <span className="dmat-focus-help">
          {busy
            ? '正在从原始事件重新聚合…'
            : onWindow
              ? '缩放用于查看；重新分箱按原始事件细化时间。'
              : '对照图沿用共同分箱与颜色尺度。'}
        </span>
      </div>
      {error && <Alert type="error" message={error} />}
    </>
  );
}

export default function DMATChart(props: Props) {
  const [open, setOpen] = useState(false);
  const focusBody = useRef<HTMLDivElement>(null);
  return (
    <div>
      <div className="dmat-preview-toolbar">
        <span className="dmat-focus-help">概览 · 滚轮滚动页面</span>
        <Button
          aria-label={`聚焦查看${props.title ? ` · ${props.title}` : ''}`}
          icon={<ExpandOutlined aria-hidden />}
          onClick={() => setOpen(true)}
        >
          聚焦查看{props.title ? ` · ${props.title}` : ''}
        </Button>
      </div>
      <DMATCanvas launch={props.launch} metric={props.metric} />
      <Modal
        title={`DMAT 聚焦查看${props.title ? ` · ${props.title}` : ''}`}
        open={open}
        onCancel={() => setOpen(false)}
        width="calc(100vw - 32px)"
        style={{ top: 16, paddingBottom: 0 }}
        className="dmat-focus-modal"
        destroyOnHidden
        afterOpenChange={(visible) => {
          if (visible)
            focusBody.current
              ?.querySelector<HTMLElement>('[aria-label="DMAT 图表操作区"]')
              ?.focus();
        }}
        footer={<Button onClick={() => setOpen(false)}>返回页面</Button>}
        styles={{
          content: { height: 'calc(100dvh - 32px)', display: 'flex', flexDirection: 'column' },
          header: { flexShrink: 0 },
          body: {
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            flex: 1,
            minHeight: 0,
            overflowY: 'auto',
          },
          footer: { flexShrink: 0 },
        }}
      >
        {open && (
          <div ref={focusBody} className="dmat-focus-body">
            <FocusedView {...props} />
          </div>
        )}
      </Modal>
    </div>
  );
}
