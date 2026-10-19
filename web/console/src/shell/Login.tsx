import Button from '../components/ActionButton';
import { useState } from 'react';
import { Alert, Input } from 'antd';
import { LockOutlined, ArrowRightOutlined } from '@ant-design/icons';
import { api } from '../api/client';
import { NodeMark } from './WorkbenchShell';

export default function Login({ onSuccess }: { onSuccess: () => void }) {
  const [token, setToken] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const login = async () => {
    setBusy(true);
    setError('');
    try {
      await api('/session/login', { token });
      setToken('');
      onSuccess();
    } catch (error) {
      setError((error as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="session-page">
      <header>
        <span className="project-mark">
          <NodeMark />
        </span>
        <strong>HQSB</strong>
        <span className="muted">/ Inference Lab</span>
      </header>
      <main className="session-form">
        <LockOutlined className="session-lock" />
        <h1>连接工作台</h1>
        <p>推理运行、分层观测与可追溯的优化证据。</p>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void login();
          }}
        >
          <label className="field-label" htmlFor="access-token">
            访问令牌
          </label>
          <Input.Password
            id="access-token"
            autoComplete="current-password"
            value={token}
            onChange={(event) => setToken(event.target.value)}
            placeholder="输入远端服务的访问令牌"
          />
          {error && <Alert className="section-gap" type="error" showIcon message={error} />}
          <Button
            htmlType="submit"
            type="primary"
            block
            loading={busy}
            disabledReason={{ reason: '请先在此页输入执行节点提供的访问令牌，再进入工作台。' }}
            disabled={!token.trim()}
            icon={<ArrowRightOutlined />}
          >
            进入工作台
          </Button>
        </form>
        <div className="session-help">
          <strong>连接到自己的执行节点</strong>
          <p>
            令牌保存在服务端私有数据目录的 access-token 文件。通过 SSH
            隧道访问，令牌不会写入浏览器存储。
          </p>
          <span>HttpOnly 会话 · 有效期 8 小时</span>
        </div>
      </main>
      <footer>HQSB · 单机真实模型观测</footer>
    </div>
  );
}
