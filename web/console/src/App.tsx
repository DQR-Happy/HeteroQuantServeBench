import Button from './components/ActionButton';
import { lazy, Suspense, useEffect, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { App as AntApp } from 'antd';
import { Navigate, Route, Routes } from 'react-router-dom';
import { api, ApiError } from './api/client';
import type { Session } from './api/types';
import { QueryState } from './components';
import Login from './shell/Login';
import WorkbenchShell from './shell/WorkbenchShell';
const Overview = lazy(() => import('./pages/Overview'));
const Benchmarks = lazy(() => import('./pages/Benchmarks'));
const Scheduling = lazy(() => import('./pages/Scheduling'));
const Precision = lazy(() => import('./pages/Precision'));
const MemoryCache = lazy(() => import('./pages/MemoryCache'));
const ModelStructure = lazy(() => import('./pages/ModelStructure'));
const Decoding = lazy(() => import('./pages/Decoding'));
const Parallel = lazy(() => import('./pages/Parallel'));
const Observations = lazy(() => import('./pages/Observations'));
const Playground = lazy(() => import('./pages/Playground'));
const Devices = lazy(() => import('./pages/Devices'));
const Machines = lazy(() => import('./pages/Machines'));
const MemoryFlow = lazy(() => import('./pages/MemoryFlow'));
const Research = lazy(() => import('./pages/Research'));
const Optimization = lazy(() => import('./pages/Optimization'));
const Runs = lazy(() => import('./pages/Runs').then((m) => ({ default: m.Runs })));
const RunDetail = lazy(() => import('./pages/Runs').then((m) => ({ default: m.RunDetail })));
const Evidence = lazy(() => import('./pages/Evidence'));
const Kernels = lazy(() => import('./pages/Evidence').then((m) => ({ default: m.KernelCatalog })));
const Compare = lazy(() => import('./pages/Compare'));
const Efficiency = lazy(() => import('./pages/Compare').then((m) => ({ default: m.Efficiency })));
const Showcase = lazy(() => import('./pages/Guide').then((m) => ({ default: m.Showcase })));
const Settings = lazy(() => import('./pages/Guide').then((m) => ({ default: m.SettingsPage })));
const UserGuide = lazy(() => import('./pages/UserGuide'));
const ModelHub = lazy(() => import('./pages/ModelHub'));
const LocalModels = lazy(() => import('./pages/LocalModels'));
const ModelArchitecture = lazy(() => import('./pages/ModelArchitecture'));

export default function App() {
  const client = useQueryClient();
  const { message } = AntApp.useApp();
  const [signedOut, setSignedOut] = useState(false);
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
  if (signedOut || (session.error instanceof ApiError && session.error.status === 401))
    return (
      <Login
        onSuccess={() => {
          setSignedOut(false);
          void client.invalidateQueries({ queryKey: ['session'] });
        }}
      />
    );
  if (session.isError)
    return (
      <div className="session-loading">
        <QueryState error={session.error} />
        <Button onClick={() => void session.refetch()}>重新连接</Button>
      </div>
    );
  if (session.isPending)
    return (
      <div className="session-loading">
        <QueryState loading />
      </div>
    );
  const logout = async () => {
    try {
      await api('/session/logout', {});
      setSignedOut(true);
      client.clear();
    } catch (error) {
      void message.error((error as Error).message);
    }
  };
  return (
    <WorkbenchShell
      version={session.data?.version}
      onLogout={session.data?.auth_required === false ? undefined : () => void logout()}
    >
      <Suspense fallback={<QueryState loading />}>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/benchmarks" element={<Benchmarks />} />
          <Route path="/scheduling" element={<Scheduling />} />
          <Route path="/precision" element={<Precision />} />
          <Route path="/memory-cache/:section?" element={<MemoryCache />} />
          <Route path="/model-structure/:section?" element={<ModelStructure />} />
          <Route path="/decoding/:section?" element={<Decoding />} />
          <Route path="/parallel/:section?" element={<Parallel />} />
          <Route path="/playground" element={<Playground />} />
          <Route path="/devices" element={<Devices />} />
          <Route path="/machines" element={<Machines />} />
          <Route path="/memory-flow" element={<MemoryFlow />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/runs/:id" element={<RunDetail />} />
          <Route path="/observations" element={<Observations />} />
          <Route path="/research" element={<Research />} />
          <Route path="/optimization" element={<Optimization />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/quantization" element={<Evidence mode="quantization" />} />
          <Route path="/kernels" element={<Kernels />} />
          <Route path="/efficiency" element={<Efficiency />} />
          <Route path="/evidence" element={<Evidence />} />
          <Route path="/experiments" element={<Evidence mode="experiments" />} />
          <Route path="/showcase" element={<Showcase />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/guide" element={<UserGuide />} />
          <Route path="/models" element={<ModelHub />} />
          <Route path="/models/local" element={<LocalModels />} />
          <Route path="/model-architecture/:section?" element={<ModelArchitecture />} />
          <Route path="*" element={<Navigate replace to="/" />} />
        </Routes>
      </Suspense>
    </WorkbenchShell>
  );
}
