import { Button, Popover, type ButtonProps } from 'antd';
import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';

export type ActionBlock = {
  reason: string;
  next?: { to: string; label: string };
};

/** Disabled controls cannot receive pointer/focus reliably. The wrapper owns
 * the explanation; the underlying action stays genuinely disabled. */
export default function ActionButton({
  disabledReason,
  ...props
}: ButtonProps & { disabledReason?: ActionBlock }) {
  const blocked = !!props.disabled || !!props.loading;
  const [open, setOpen] = useState(false);
  const trigger = useRef<HTMLSpanElement>(null);
  const content = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!blocked) setOpen(false);
  }, [blocked]);
  if (!blocked) return <Button {...props} />;
  const explanation = disabledReason ?? {
    reason: '此操作正在提交或执行，请等待当前操作完成后再试。',
  };
  return (
    <Popover
      title="暂不可执行"
      open={open}
      onOpenChange={(value) => {
        if (
          !value &&
          (trigger.current?.contains(document.activeElement) ||
            content.current?.contains(document.activeElement))
        )
          return;
        setOpen(value);
      }}
      trigger={['hover', 'click']}
      mouseEnterDelay={0.12}
      content={
        <div
          ref={content}
          className="action-explanation"
          onBlur={(event) => {
            if (
              !trigger.current?.contains(event.relatedTarget) &&
              !content.current?.contains(event.relatedTarget)
            )
              setOpen(false);
          }}
          onKeyDown={(event) => {
            if (event.key === 'Escape') {
              trigger.current?.focus();
              setOpen(false);
            }
          }}
        >
          <p>{explanation.reason}</p>
          {explanation.next && <Link to={explanation.next.to}>{explanation.next.label} →</Link>}
        </div>
      }
    >
      <span
        ref={trigger}
        className={`disabled-action ${props.block ? 'disabled-action-block' : ''}`}
        tabIndex={0}
        data-disabled-action
        aria-label={`不可用原因：${explanation.reason}`}
        onFocus={() => setOpen(true)}
        onBlur={(event) => {
          if (
            !content.current?.contains(event.relatedTarget) &&
            !trigger.current?.contains(event.relatedTarget)
          )
            setOpen(false);
        }}
        onKeyDown={(event) => {
          if (event.key === 'Escape') setOpen(false);
        }}
      >
        <Button {...props} disabled style={{ ...props.style, pointerEvents: 'none' }} />
      </span>
    </Popover>
  );
}
