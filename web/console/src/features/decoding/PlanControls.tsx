import { Input, InputNumber, Select, Space } from 'antd';
import { defaults, type DecodePlan } from './types';
export default function PlanControls({
  plan,
  onChange,
  choices,
  onChoices,
  drafts,
}: {
  plan: DecodePlan;
  onChange: (p: DecodePlan) => void;
  choices: string;
  onChoices: (s: string) => void;
  drafts: { id: string; name: string }[];
}) {
  const set = (key: keyof DecodePlan, value: unknown) => onChange({ ...plan, [key]: value });
  return (
    <>
      <Space wrap>
        <label>
          生成策略{' '}
          <Select
            aria-label="L7 策略"
            style={{ width: 280 }}
            value={plan.method}
            onChange={(method) => {
              onChange({ ...defaults, method });
              onChoices('');
            }}
            options={[
              { value: 'autoregressive', label: '逐 Token / 贪心或采样' },
              { value: 'prompt_lookup', label: '提示词查找推测 / 贪心验证' },
              { value: 'assistant', label: '独立草稿模型推测 / 贪心验证' },
            ]}
          />
        </label>
        {plan.method === 'autoregressive' ? (
          <label>
            Token 选择{' '}
            <Select
              aria-label="L7 Token 选择"
              style={{ width: 170 }}
              value={plan.sampling}
              onChange={(sampling) =>
                onChange({
                  ...plan,
                  sampling,
                  temperature: sampling === 'greedy' ? 1 : 0.8,
                  top_k: sampling === 'greedy' ? 0 : 40,
                  top_p: sampling === 'greedy' ? 1 : 0.95,
                })
              }
              options={[
                { value: 'greedy', label: 'Greedy / 最大概率' },
                { value: 'multinomial', label: 'Multinomial / 随机采样' },
              ]}
            />
          </label>
        ) : (
          <label>
            每轮草稿预算{' '}
            <InputNumber
              aria-label="L7 草稿预算"
              min={1}
              max={16}
              value={plan.draft_tokens}
              onChange={(v) => set('draft_tokens', v ?? 4)}
            />
          </label>
        )}
        {plan.method === 'assistant' && (
          <label>
            草稿模型{' '}
            <Select
              aria-label="L7 草稿模型"
              showSearch
              optionFilterProp="label"
              style={{ width: 330 }}
              placeholder="选择已下载并登记的模型"
              value={plan.draft_deployment_id}
              onChange={(v) => set('draft_deployment_id', v)}
              options={drafts.map((d) => ({ value: d.id, label: d.name }))}
            />
          </label>
        )}
      </Space>
      {plan.method === 'autoregressive' && (
        <>
          <Space wrap>
            {plan.sampling === 'multinomial' && (
              <>
                <label>
                  Temperature{' '}
                  <InputNumber
                    aria-label="Temperature"
                    min={0.05}
                    max={5}
                    step={0.05}
                    value={plan.temperature}
                    onChange={(v) => set('temperature', v ?? 1)}
                  />
                </label>
                <label>
                  Top-k（0 为关闭）{' '}
                  <InputNumber
                    aria-label="Top-k"
                    min={0}
                    max={1024}
                    value={plan.top_k}
                    onChange={(v) => set('top_k', v ?? 0)}
                  />
                </label>
                <label>
                  Top-p{' '}
                  <InputNumber
                    aria-label="Top-p"
                    min={0.01}
                    max={1}
                    step={0.05}
                    value={plan.top_p}
                    onChange={(v) => set('top_p', v ?? 1)}
                  />
                </label>
                <label>
                  Seed{' '}
                  <InputNumber
                    aria-label="L7 Seed"
                    min={0}
                    max={2147483647}
                    value={plan.seed}
                    onChange={(v) => set('seed', v ?? 42)}
                  />
                </label>
              </>
            )}
            <label>
              重复惩罚{' '}
              <InputNumber
                aria-label="重复惩罚"
                min={1}
                max={2}
                step={0.05}
                value={plan.repetition_penalty}
                onChange={(v) => set('repetition_penalty', v ?? 1)}
              />
            </label>
            <label>
              最少生成 Token{' '}
              <InputNumber
                aria-label="最少生成 Token"
                min={0}
                max={256}
                value={plan.min_new_tokens}
                onChange={(v) => set('min_new_tokens', v ?? 0)}
              />
            </label>
          </Space>
          <label className="field-label">有限候选（可选，每行一个完整回答）</label>
          <Input.TextArea
            aria-label="L7 有限候选"
            rows={3}
            value={choices}
            onChange={(e) => {
              onChoices(e.target.value);
              if (!e.target.value.trim()) set('constraint_backend', 'gather');
            }}
            placeholder="留空表示不限制；例：positive / negative / neutral"
          />
          {choices.trim() && (
            <label>
              约束选择实现{' '}
              <Select
                aria-label="L7 约束实现"
                value={plan.constraint_backend}
                onChange={(v) => set('constraint_backend', v)}
                style={{ width: 280 }}
                options={[
                  { value: 'mask', label: '全词表 Mask 基线' },
                  { value: 'gather', label: '候选 Token Gather' },
                ]}
              />
            </label>
          )}
          <p>
            处理顺序：重复惩罚 → 最少长度/EOS 与候选限制 → 温度 → Top-k → Top-p →
            选择。有限候选需要取消固定长度，并预留最长候选加 EOS 的预算；不是任意 JSON Schema。
          </p>
        </>
      )}
      {plan.method === 'prompt_lookup' && (
        <p>
          从已提交的输入和生成历史中查找重复
          n-gram，提出后续候选，交给目标模型一次批量验证。无匹配时回到单 Token
          目标验证，仍会如实记录零接受率。
        </p>
      )}
      {plan.method === 'assistant' && (
        <p>
          首个请求显式触发加载，之后草稿权重驻留在当前
          worker，直到草稿或目标模型被卸载。需要完全相同的 Tokenizer 编码与 ID
          映射；本轮校验不匹配会失败，不静默切换算法。
        </p>
      )}
    </>
  );
}
