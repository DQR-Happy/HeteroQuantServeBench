import Button from '../components/ActionButton';
import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Form, Input, InputNumber, Select, Space, Table, Tag } from 'antd';
import { api, selectedMachine } from '../api/client';
import { PageHead, Panel, QueryState } from '../components';
import { activateMachine } from '../features/machines/types';
import type { Machine, Machines as MachineList } from '../features/machines/types';

export default function Machines() {
  const client = useQueryClient();
  const [form] = Form.useForm<Machine>();
  const transport = Form.useWatch('transport', form) ?? 'ssh';
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [checks, setChecks] = useState<Record<string, string>>({});
  const query = useQuery({ queryKey: ['machines'], queryFn: () => api<MachineList>('/machines') });
  const act = async (id: string, action: 'probe' | 'activate' | 'remove') => {
    setBusy(id);
    setError('');
    try {
      if (action === 'activate') {
        await activateMachine(id);
        return;
      }
      await api(`/machines/${encodeURIComponent(id)}/${action}`, {});
      if (action === 'probe') setChecks((old) => ({ ...old, [id]: '连接与 API 已验证' }));
      else await client.invalidateQueries({ queryKey: ['machines'] });
    } catch (failure) {
      setError((failure as Error).message);
      if (action === 'probe') setChecks((old) => ({ ...old, [id]: '连接失败' }));
    } finally {
      setBusy('');
    }
  };
  const save = async (values: Machine) => {
    setBusy('new');
    setError('');
    try {
      const common = {
        id: values.id,
        name: values.name,
        transport: values.transport,
        ...(values.token_env ? { token_env: values.token_env } : {}),
      };
      const config =
        values.transport === 'http'
          ? { ...common, base_url: values.base_url }
          : {
              ...common,
              ssh_host: values.ssh_host,
              remote_port: values.remote_port ?? 8768,
              ...(values.ssh_user ? { ssh_user: values.ssh_user } : {}),
              ...(values.ssh_port ? { ssh_port: values.ssh_port } : {}),
              ...(values.identity_file ? { identity_file: values.identity_file } : {}),
            };
      await api('/machines', config);
      form.resetFields();
      await client.invalidateQueries({ queryKey: ['machines'] });
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setBusy('');
    }
  };
  return (
    <>
      <PageHead
        kicker="MACHINE CONNECTIONS"
        title="后端机器与连接"
        description="保存连接、测试可达性并切换当前工作台。切换后部署、运行、评测和制品下载都来自所选节点。"
      />
      <Alert
        className="section-gap"
        showIcon
        type="info"
        message="任务与证据留在原机器"
        description="切换仅改变本标签页的工作台目标，已有任务继续执行；不会搬运模型或合并记录。SSH 与网络连接都要求目标端已启动 HQSB 服务。"
      />
      <Panel title="已保存连接" subtitle="先测试连接，再切换。连接失败时仍可在此返回本机。">
        <QueryState loading={query.isPending} error={query.error} />
        <Table<Machine>
          rowKey="id"
          size="small"
          pagination={false}
          scroll={{ x: 850 }}
          dataSource={query.data?.items}
          columns={[
            {
              title: '节点',
              render: (_, row) => (
                <>
                  <strong>{row.name}</strong> {row.id === selectedMachine && <Tag>当前</Tag>}
                  <div className="help-text mono">{row.id}</div>
                </>
              ),
            },
            {
              title: '连接',
              render: (_, row) =>
                row.transport === 'local'
                  ? '控制服务所在机器'
                  : row.transport === 'http'
                    ? row.base_url
                    : `SSH ${row.ssh_user ? row.ssh_user + '@' : ''}${row.ssh_host}${row.ssh_port ? ':' + row.ssh_port : ''} → 127.0.0.1:${row.remote_port}`,
            },
            { title: '最近测试', render: (_, row) => checks[row.id] ?? '未测试' },
            {
              title: '操作',
              render: (_, row) => (
                <Space wrap>
                  <Button
                    size="small"
                    disabledReason={{ reason: '正在处理另一项机器连接操作，请等待完成后再测试。' }}
                    disabled={!!busy}
                    onClick={() => void act(row.id, 'probe')}
                  >
                    测试连接
                  </Button>
                  <Button
                    size="small"
                    disabledReason={{
                      reason: busy
                        ? '正在处理机器连接操作，请等待完成。'
                        : '这已经是当前使用的机器，无需重复切换。',
                    }}
                    disabled={!!busy || row.id === selectedMachine}
                    onClick={() => void act(row.id, 'activate')}
                  >
                    切换到此机器
                  </Button>
                  {row.transport !== 'local' && (
                    <Button
                      size="small"
                      disabledReason={{
                        reason: busy
                          ? '正在处理机器连接操作，请等待完成。'
                          : '当前使用的连接不能移除。请先切换到其他机器，再返回移除此连接。',
                        next: { to: '/machines', label: '选择并切换其他机器' },
                      }}
                      disabled={!!busy || row.id === selectedMachine}
                      onClick={() => void act(row.id, 'remove')}
                    >
                      移除连接
                    </Button>
                  )}
                </Space>
              ),
            },
          ]}
        />
      </Panel>
      {error && <Alert className="section-gap" showIcon type="error" message={error} />}
      <Panel
        title="新增机器"
        subtitle="连接配置保存在控制服务的数据目录，私钥与令牌内容不经过浏览器。"
      >
        <Form
          form={form}
          layout="vertical"
          initialValues={{ transport: 'ssh', remote_port: 8768 }}
          onFinish={(values) => void save(values)}
        >
          <div className="grid-two">
            <Form.Item
              name="id"
              label="连接 ID"
              rules={[
                {
                  required: true,
                  pattern: /^[a-z0-9][a-z0-9_-]{0,63}$/,
                  message: '使用小写字母、数字、下划线或连字符',
                },
              ]}
            >
              <Input placeholder="例如 4090d-ssh" />
            </Form.Item>
            <Form.Item name="name" label="显示名称" rules={[{ required: true }]}>
              <Input placeholder="例如 RTX 4090D" />
            </Form.Item>
          </div>
          <Form.Item name="transport" label="连接方式">
            <Select
              options={[
                { value: 'ssh', label: 'SSH 隧道' },
                { value: 'http', label: 'HTTP / HTTPS 网络连接' },
              ]}
            />
          </Form.Item>
          {transport === 'ssh' ? (
            <>
              <Form.Item name="ssh_host" label="SSH 主机或别名" rules={[{ required: true }]}>
                <Input placeholder="控制服务机器上的 SSH config 别名，或主机地址" />
              </Form.Item>
              <div className="grid-two">
                <Form.Item name="ssh_user" label="用户名（别名可省略）">
                  <Input />
                </Form.Item>
                <Form.Item name="ssh_port" label="SSH 端口（别名可省略）">
                  <InputNumber min={1} max={65535} />
                </Form.Item>
                <Form.Item
                  name="remote_port"
                  label="目标 HQSB 服务端口"
                  rules={[{ required: true }]}
                >
                  <InputNumber min={1} max={65535} />
                </Form.Item>
                <Form.Item name="identity_file" label="私钥路径（位于控制服务机器，可省略）">
                  <Input placeholder="使用目标服务进程可读的私钥或 SSH agent" />
                </Form.Item>
              </div>
              <p className="help-text">
                SSH 主机密钥须已在控制机 known_hosts 中核验。Mac 的 SSH config
                和私钥不会自动复制到远端；连接错误会保留原因类别，不会自动下载代码或安装环境。
              </p>
            </>
          ) : (
            <Form.Item name="base_url" label="HQSB 服务地址" rules={[{ required: true }]}>
              <Input placeholder="https://node.example.com:8768（不包含 /api 路径）" />
            </Form.Item>
          )}
          <Form.Item name="token_env" label="目标服务访问令牌的环境变量名（免登录服务留空）">
            <Input placeholder="例如 HQSB_REMOTE_TOKEN，仅填写变量名" />
          </Form.Item>
          <Button
            htmlType="submit"
            type="primary"
            loading={busy === 'new'}
            disabledReason={{ reason: '正在保存、测试或切换连接，请等待当前操作完成后再保存。' }}
            disabled={!!busy}
          >
            保存连接
          </Button>
        </Form>
      </Panel>
    </>
  );
}
