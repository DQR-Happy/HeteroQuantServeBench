import Button from '../components/ActionButton';
import { useEffect, useState, type ReactNode } from 'react';
import { Drawer, Input, Modal, Tooltip } from 'antd';
import {
  ArrowRightOutlined,
  LogoutOutlined,
  MenuOutlined,
  SearchOutlined,
  SettingOutlined,
  ReadOutlined,
} from '@ant-design/icons';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { destinations, navigation } from './navigation';
import MachineSwitch from '../features/machines/MachineSwitch';

export default function WorkbenchShell({
  children,
  version,
  onLogout,
}: {
  children: ReactNode;
  version?: string;
  onLogout?: () => void;
}) {
  const location = useLocation();
  const navigate = useNavigate();
  const path = location.pathname.startsWith('/runs/')
    ? '/runs'
    : location.pathname.startsWith('/decoding/')
      ? '/decoding'
      : location.pathname.startsWith('/parallel/')
        ? '/parallel'
        : location.pathname.startsWith('/model-structure/')
          ? '/model-structure'
          : location.pathname.startsWith('/memory-cache/')
            ? '/memory-cache'
            : location.pathname.startsWith('/model-architecture/')
              ? '/model-architecture'
              : location.pathname;
  const guideChapter =
    (
      {
        '/benchmarks': 'benchmarks',
        '/playground': 'playground',
        '/machines': 'machines',
        '/models': 'models',
        '/models/local': 'models',
        '/devices': 'machines',
        '/optimization': 'experiments',
        '/model-structure': 'model-structure',
        '/decoding': 'decoding',
        '/parallel': 'parallel',
        '/memory-cache': 'memory-cache',
        '/observations': 'observations',
        '/memory-flow': 'observations',
        '/research': 'observations',
        '/runs': 'evidence',
        '/compare': 'evidence',
        '/evidence': 'evidence',
        '/experiments': 'evidence',
        '/kernels': 'kernels',
        '/quantization': 'boundaries',
        '/efficiency': 'evidence',
      } as Record<string, string>
    )[path] ?? 'start';
  const group =
    navigation.find((group) => group.pages.some((page) => page.path === path)) ?? navigation[0];
  const [menuOpen, setMenuOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [search, setSearch] = useState('');
  const matches = destinations.filter((item) =>
    `${item.label} ${item.detail} ${item.path}`.toLowerCase().includes(search.toLowerCase()),
  );
  useEffect(() => {
    setMenuOpen(false);
    setSearchOpen(false);
  }, [location.pathname]);
  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (event.isComposing || target.matches('input,textarea,[contenteditable="true"]')) return;
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        setSearch('');
        setSearchOpen((value) => !value);
      }
    };
    window.addEventListener('keydown', listener);
    return () => window.removeEventListener('keydown', listener);
  }, []);
  const pageNav = (mobile = false) => (
    <nav
      aria-label={mobile ? '全部页面' : '项目导航'}
      className={mobile ? 'drawer-links' : 'project-tabs'}
    >
      {navigation.map((item) =>
        mobile ? (
          <section key={item.id}>
            <h3>{item.label}</h3>
            {item.pages.map((page) => (
              <Link
                key={page.path}
                to={page.path}
                aria-current={path === page.path ? 'page' : undefined}
              >
                {page.label}
              </Link>
            ))}
          </section>
        ) : (
          <Link
            key={item.id}
            to={item.path}
            aria-current={group.id === item.id ? 'page' : undefined}
          >
            {item.icon}
            <span>{item.label}</span>
          </Link>
        ),
      )}
    </nav>
  );
  return (
    <div className="workbench-shell">
      <a className="skip-link" href="#main-content">
        跳到主内容
      </a>
      <header className="project-header">
        <div className="project-bar">
          <Button
            className="mobile-menu"
            type="text"
            icon={<MenuOutlined />}
            aria-label="打开导航"
            onClick={() => setMenuOpen(true)}
          />
          <Link to="/" className="project-mark" aria-label="HQSB 工作空间">
            <NodeMark />
          </Link>
          <div className="project-identity">
            <span className="project-owner">HQSB</span>
            <span className="path-divider">/</span>
            <Link to="/">Inference Lab</Link>
            <span className="scope-label">单机观测</span>
          </div>
          <div className="project-tools">
            <MachineSwitch />
            <Link className="manual-entry" to={`/guide#${guideChapter}`} aria-label="使用指南">
              <ReadOutlined /> <span>使用指南</span>
            </Link>
            <button
              className="quick-open"
              onClick={() => {
                setSearch('');
                setSearchOpen(true);
              }}
              aria-label="快速跳转"
            >
              <SearchOutlined />
              <span>跳转页面…</span>
              <kbd>⌘ / Ctrl K</kbd>
            </button>
            <Tooltip title="设置与使用指引">
              <Link className="icon-link" to="/settings" aria-label="设置与帮助">
                <SettingOutlined />
              </Link>
            </Tooltip>
            {onLogout && (
              <Tooltip title="退出会话；执行中的任务继续运行">
                <Button
                  type="text"
                  aria-label="退出会话"
                  icon={<LogoutOutlined />}
                  onClick={onLogout}
                />
              </Tooltip>
            )}
          </div>
        </div>
        {pageNav()}
      </header>
      <div className="context-bar">
        <nav aria-label={`${group.label}页面`}>
          {group.pages.map((page) => (
            <Link
              to={page.path}
              key={page.path}
              aria-current={path === page.path ? 'page' : undefined}
            >
              {page.label}
            </Link>
          ))}
        </nav>
        <span className="session-indicator">
          <span className="live-dot" />
          {onLogout ? '私有会话' : '已连接'} <span className="muted">· 服务 {version ?? '—'}</span>
        </span>
      </div>
      <main id="main-content" tabIndex={-1} className={`main-content area-${group.id}`}>
        {children}
      </main>
      <footer className="workbench-footer">
        <span>
          HQSB <span className="muted">/ Inference observability</span>
        </span>
        <span>
          UI 0.4 <span className="footer-separator">·</span> 单次诊断与正式验收分开计量
        </span>
      </footer>
      <Drawer
        title="工作空间导航"
        placement="left"
        width={300}
        open={menuOpen}
        onClose={() => setMenuOpen(false)}
      >
        {pageNav(true)}
      </Drawer>
      <Modal
        title="跳转到工作区"
        open={searchOpen}
        onCancel={() => setSearchOpen(false)}
        footer={null}
        className="jump-dialog"
        afterOpenChange={(open) => {
          if (open) document.getElementById('jump-query')?.focus();
        }}
      >
        <Input
          id="jump-query"
          aria-label="搜索工作区"
          prefix={<SearchOutlined />}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="页面、功能或路径"
          allowClear
          onPressEnter={() => {
            if (matches[0]) {
              navigate(matches[0].path);
              setSearchOpen(false);
            }
          }}
        />
        <div className="jump-results">
          {matches.length ? (
            matches.map((item) => (
              <Link to={item.path} key={item.path} onClick={() => setSearchOpen(false)}>
                {item.icon}
                <span>
                  <strong>{item.label}</strong>
                  <small>
                    {item.group} · {item.detail}
                  </small>
                </span>
                <ArrowRightOutlined />
              </Link>
            ))
          ) : (
            <p className="empty-inline">没有匹配的工作区。尝试“内存”“运行”或“质量”。</p>
          )}
        </div>
      </Modal>
    </div>
  );
}
export function NodeMark() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M5 5v14M19 5v14M5 12h14" stroke="currentColor" strokeWidth="2" />
      <path d="M9 8h6v8H9z" fill="currentColor" />
      <path d="M2 5h6M16 19h6" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}
