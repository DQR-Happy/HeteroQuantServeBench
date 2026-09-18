import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { App, Select } from 'antd';
import { Link } from 'react-router-dom';
import { api, selectedMachine } from '../../api/client';
import { activateMachine } from './types';
import type { Machines } from './types';

export default function MachineSwitch() {
  const { message } = App.useApp();
  const [busy, setBusy] = useState(false);
  const query = useQuery({
    queryKey: ['machines'],
    queryFn: () => api<Machines>('/machines'),
    retry: false,
  });
  return (
    <div className="machine-switch">
      <Select
        aria-label="当前后端机器"
        value={selectedMachine}
        style={{ width: 210 }}
        loading={busy || query.isPending}
        disabled={busy}
        options={query.data?.items.map((row) => ({ value: row.id, label: row.name }))}
        onChange={async (id) => {
          setBusy(true);
          try {
            await activateMachine(id);
          } catch (error) {
            void message.error((error as Error).message);
            setBusy(false);
          }
        }}
      />
      <Link to="/machines">管理连接</Link>
    </div>
  );
}
