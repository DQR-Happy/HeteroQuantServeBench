import Button from '../../components/ActionButton';
import { useState } from 'react';
import { InputNumber, Select, Space } from 'antd';

const presets = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

export default function OutputTokenBudget({
  id,
  value,
  onChange,
  maximum,
  minimum = 1,
  disabled,
  invalid,
  describedBy,
}: {
  id: string;
  value: number | null;
  onChange: (value: number | null) => void;
  maximum?: number;
  minimum?: number;
  disabled?: boolean;
  invalid?: boolean;
  describedBy?: string;
}) {
  const [custom, setCustom] = useState(false);
  const available = presets.filter(
    (item) => item >= minimum && (maximum == null || item < maximum),
  );
  const selected =
    custom || value == null
      ? 'custom'
      : value === maximum
        ? 'maximum'
        : available.includes(value)
          ? String(value)
          : 'custom';
  const useMaximum = () => {
    if (maximum == null || maximum < minimum) return;
    setCustom(false);
    onChange(maximum);
  };
  return (
    <>
      <Space.Compact style={{ width: '100%' }}>
        <Select
          id={id}
          value={selected}
          style={{ flex: 1, minWidth: 0 }}
          disabled={disabled}
          status={invalid ? 'error' : undefined}
          aria-invalid={!!invalid}
          aria-describedby={describedBy}
          options={[
            ...available.map((item) => ({ value: String(item), label: `${item} tokens` })),
            {
              value: 'maximum',
              label: `最大值 · ${maximum ?? '未知'} tokens`,
              disabled: maximum == null || maximum < minimum,
            },
            { value: 'custom', label: '自定义' },
          ]}
          onChange={(choice) => {
            if (choice === 'maximum') useMaximum();
            else if (choice === 'custom') setCustom(true);
            else {
              setCustom(false);
              onChange(Number(choice));
            }
          }}
        />
        <Button
          onClick={useMaximum}
          disabledReason={{
            reason: disabled
              ? '当前任务正在执行，输出预算已冻结，请等待完成后再调整。'
              : maximum == null
                ? '尚未取得可用输出额度，请先加载模型并填写输入文本，等待 token 预算检查完成。'
                : '当前输入已占满可用窗口，请缩短输入文本。',
            next: { to: '/playground', label: '检查模型、输入与输出预算' },
          }}
          disabled={disabled || maximum == null || maximum < minimum}
        >
          最大值
        </Button>
      </Space.Compact>
      {selected === 'custom' && (
        <InputNumber
          aria-label="自定义输出 token 数"
          value={value}
          onChange={onChange}
          status={invalid ? 'error' : undefined}
          aria-invalid={!!invalid}
          aria-describedby={describedBy}
          disabled={disabled}
          style={{ width: '100%', marginTop: 8 }}
        />
      )}
    </>
  );
}
