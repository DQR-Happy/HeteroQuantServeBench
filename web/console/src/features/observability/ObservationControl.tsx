import { Alert, Select } from 'antd';
import type { ObservationMode } from './types';

export default function ObservationControl({
  value,
  onChange,
  disabled,
  local,
}: {
  value: ObservationMode;
  onChange: (value: ObservationMode) => void;
  disabled: boolean;
  local: boolean;
}) {
  return (
    <div className="observation-control">
      <label className="field-label" htmlFor="observation-mode">
        观测模式
      </label>
      <Select
        id="observation-mode"
        value={local ? value : 'off'}
        onChange={onChange}
        disabled={disabled || !local}
        className="wide-select"
        options={[
          { value: 'off', label: '关闭详录 · 仅请求指标' },
          { value: 'basic', label: '基础观测 · 阶段 / token / 内存' },
          { value: 'operators', label: '算子诊断 · PyTorch Profiler' },
        ]}
      />
      <div className="help-text">
        {local
          ? '设置只影响下一次提交。基础观测记录工作进程的主机时间；不能等同 GPU kernel 执行时间。'
          : '外部兼容 API 未提供引擎内采集，仅保留请求层指标。'}
      </div>
      {local && value === 'operators' && (
        <Alert
          type="warning"
          showIcon
          message="诊断采集有开销"
          description="窗口最多覆盖 prefill 和前 8 个 decode step。记录 shape 与内存并导出 trace；采集结果不作为无插桩速度基准，设备事件缺失会明确显示。"
        />
      )}
    </div>
  );
}
