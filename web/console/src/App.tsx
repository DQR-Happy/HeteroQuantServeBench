import { lazy, Suspense, useEffect, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, App as AntApp, Button, Input, Layout, Menu, Spin, Tag, Tooltip } from 'antd';
import {
  DashboardOutlined,
  ThunderboltOutlined,
  DeploymentUnitOutlined,
  UnorderedListOutlined,
  ExperimentOutlined,
  CodeOutlined,
  FileSearchOutlined,
  BarChartOutlined,
  SwapOutlined,
  PlayCircleOutlined,
  SettingOutlined,
  LogoutOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
} from '@ant-design/icons';
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { api } from './api/client';
import type { Session } from './api/types';
import { QueryState } from './components';
const Overview = lazy(() => import('./pages/Overview'));
const Playground = lazy(() => import('./pages/Playground'));
const Devices = lazy(() => import('./pages/Devices'));
const MemoryFlow = lazy(() => import('./pages/MemoryFlow'));
const Research = lazy(() => import('./pages/Research'));
const Runs = lazy(() => import('./pages/Runs').then((m) => ({ default: m.Runs })));
const RunDetail = lazy(() => import('./pages/Runs').then((m) => ({ default: m.RunDetail })));
const Evidence = lazy(() => import('./pages/Evidence'));
const Kernels = lazy(() => import('./pages/Evidence').then((m) => ({ default: m.KernelCatalog })));
const Compare = lazy(() => import('./pages/Compare'));
const Efficiency = lazy(() => import('./pages/Compare').then((m) => ({ default: m.Efficiency })));
const Showcase = lazy(() => import('./pages/Guide').then((m) => ({ default: m.Showcase })));
const Settings = lazy(() => import('./pages/Guide').then((m) => ({ default: m.SettingsPage })));

function Login({ onSuccess }: { onSuccess: () => void }) {
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
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="login-page">
      <div className="login-story">
        <Link className="brand" to="/">
          <span className="brand-symbol">H</span>
          <strong>
            HQSB<span> CONSOLE</span>
          </strong>
        </Link>
        <div>
          <div className="eyebrow">HETEROGENEOUS INFERENCE LAB</div>
          <h1>
            让优化，
            <br />
            有迹可循。
          </h1>
          <p>
            从真实请求到实验证据，
            <br />
            连接模型、算子与异构算力。
          </p>
        </div>
        <span className="small muted">OBSERVE · COMPARE · EXPLAIN</span>
      </div>
      <main className="login-form">
        <Tag color="green">私有工作台</Tag>
        <h2>连接你的推理实验室</h2>
        <p className="muted">
          输入服务启动时生成的访问令牌。令牌保存在远端私有数据目录，不会写入浏览器存储。
        </p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void login();
          }}
        >
          <label className="field-label" htmlFor="access-token">
            访问令牌
          </label>
          <Input.Password
            id="access-token"
            autoComplete="current-password"
            size="large"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="HQSB Console access token"
          />
          {error && <Alert className="section-gap" type="error" showIcon message={error} />}
          <Button
            htmlType="submit"
            type="primary"
            size="large"
            block
            loading={busy}
            disabled={!token.trim()}
          >
            进入工作台
          </Button>
        </form>
        <div className="help-text">
          默认路径：.console/access-token
          <br />
          通过 SSH 隧道访问；会话有效期 8 小时。
        </div>
      </main>
    </div>
  );
}

export default function App() {
  const client = useQueryClient();
  const navigate = useNavigate();
  const location = useLocation();
  const { message } = AntApp.useApp();
  const [signedOut, setSignedOut] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const session = useQuery({
    queryKey: ['session'],
    queryFn: () => api<Session>('/session'),
    retry: false,
    enabled: !signedOut,
  });
  useEffect(() => {
    const listener = () => {
      setSignedOut(true);
      client.clear();
    };
    window.addEventListener('hqsb:unauth', listener);
    return () => window.removeEventListener('hqsb:unauth', listener);
  }, [client]);
  if (signedOut || session.isError)
    return (
      <Login
        onSuccess={() => {
          setSignedOut(false);
          void client.invalidateQueries({ queryKey: ['session'] });
        }}
      />
    );
  if (session.isPending)
    return (
      <div className="loading full-height">
        <Spin size="large" />
        <span>连接 HQSB Console…</span>
      </div>
    );
  const items = [
    { key: '/', icon: <DashboardOutlined />, label: '总览' },
    { key: '/playground', icon: <ThunderboltOutlined />, label: '推理工作台' },
    { key: '/devices', icon: <DeploymentUnitOutlined />, label: '设备与部署' },
    { key: '/memory-flow', icon: <SwapOutlined />, label: '内存与数据流' },
    { key: '/runs', icon: <UnorderedListOutlined />, label: '运行记录' },
    { key: '/research', icon: <FileSearchOutlined />, label: '深度分析' },
    { type: 'divider' as const },
    { key: '/compare', icon: <SwapOutlined />, label: '运行对比' },
    { key: '/quantization', icon: <ExperimentOutlined />, label: '量化与质量' },
    { key: '/kernels', icon: <CodeOutlined />, label: '算子与配置' },
    { key: '/efficiency', icon: <BarChartOutlined />, label: '性能与能效' },
    { key: '/evidence', icon: <FileSearchOutlined />, label: '证据中心' },
    { key: '/experiments', icon: <ExperimentOutlined />, label: '实验地图' },
    { type: 'divider' as const },
    { key: '/showcase', icon: <PlayCircleOutlined />, label: '技术展示' },
    { key: '/settings', icon: <SettingOutlined />, label: '设置与帮助' },
  ];
  const current = location.pathname.startsWith('/runs/') ? '/runs' : location.pathname;
  const logout = async () => {
    try {
      await api('/session/logout', {});
      setSignedOut(true);
      client.clear();
    } catch (e) {
      void message.error((e as Error).message);
    }
  };
  return (
    <Layout className="app-layout">
      <a className="skip-link" href="#main-content">
        跳到主内容
      </a>
      <Layout.Sider
        className="sidebar"
        width={232}
        collapsedWidth={76}
        collapsed={collapsed}
        breakpoint="lg"
        onBreakpoint={setCollapsed}
      >
        <Link className="brand" to="/">
          <span className="brand-symbol">H</span>
          {!collapsed && (
            <strong>
              HQSB<small>INFERENCE CONSOLE</small>
            </strong>
          )}
        </Link>
        {!collapsed && <div className="sidebar-label">WORKSPACE</div>}
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[current]}
          items={items}
          onClick={(e) => navigate(e.key)}
        />
        <div className="sidebar-foot">
          <span className="live-dot" />
          {!collapsed && (
            <span>
              真实执行 · 证据可追溯<small>CONSOLE v{session.data?.version}</small>
            </span>
          )}
        </div>
      </Layout.Sider>
      <Layout className="main-layout" style={{ marginLeft: collapsed ? 76 : 232 }}>
        <header className="topbar">
          <div>
            <Button
              type="text"
              aria-label="折叠菜单"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed((v) => !v)}
            />
            <span className="breadcrumb">
              HQSB / <strong>{items.find((i) => i.key === current)?.label ?? '运行详情'}</strong>
            </span>
          </div>
          <div className="topbar-right">
            <span className="small muted">HeteroQuantServeBench</span>
            <Tag className="version-tag">UI v0.2.1</Tag>
            <Tag className="live-tag">
              <span className="live-dot" /> LIVE
            </Tag>
            <Tooltip title="退出会话；正在执行的请求不会自动取消">
              <Button
                type="text"
                aria-label="退出会话"
                icon={<LogoutOutlined />}
                onClick={() => void logout()}
              />
            </Tooltip>
          </div>
        </header>
        <main id="main-content" className="main-content">
          <Suspense fallback={<QueryState loading />}>
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/playground" element={<Playground />} />
              <Route path="/devices" element={<Devices />} />
              <Route path="/memory-flow" element={<MemoryFlow />} />
              <Route path="/runs" element={<Runs />} />
              <Route path="/runs/:id" element={<RunDetail />} />
              <Route path="/research" element={<Research />} />
              <Route path="/compare" element={<Compare />} />
              <Route path="/quantization" element={<Evidence mode="quantization" />} />
              <Route path="/kernels" element={<Kernels />} />
              <Route path="/efficiency" element={<Efficiency />} />
              <Route path="/evidence" element={<Evidence />} />
              <Route path="/experiments" element={<Evidence mode="experiments" />} />
              <Route path="/showcase" element={<Showcase />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="*" element={<Navigate replace to="/" />} />
            </Routes>
          </Suspense>
        </main>
        <footer className="app-footer">
          HQSB · Every optimization needs evidence.<span>交互诊断 ≠ 正式性能认证</span>
        </footer>
      </Layout>
    </Layout>
  );
}
