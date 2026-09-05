import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { App as AntApp, ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import App from './App';
import './styles.css';

const client = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 1500, refetchOnWindowFocus: false },
    mutations: { retry: false },
  },
});
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ConfigProvider
      locale={zhCN}
      theme={{
        token: {
          colorPrimary: '#295c96',
          colorInfo: '#295c96',
          colorSuccess: '#217345',
          colorWarning: '#805b10',
          colorError: '#b42332',
          colorInfoBg: '#eef3f8',
          colorInfoBorder: '#c4d1e2',
          colorSuccessBg: '#eef5ef',
          colorSuccessBorder: '#c4d9cc',
          colorWarningBg: '#faf6e8',
          colorWarningBorder: '#e3d8b6',
          colorErrorBg: '#fff1f0',
          colorErrorBorder: '#ecc7c8',
          colorText: '#1f2328',
          colorTextSecondary: '#59636e',
          colorBorder: '#c2c9d1',
          borderRadius: 6,
          fontSize: 14,
          controlHeight: 32,
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif',
        },
        components: {
          Menu: {
            itemBg: 'transparent',
            itemSelectedBg: '#dfe8f2',
            itemSelectedColor: '#304e72',
            itemHeight: 36,
          },
          Table: { headerBg: '#f6f8fa', cellPaddingBlock: 12, cellPaddingInline: 16 },
          Button: { primaryShadow: 'none' },
        },
      }}
    >
      <AntApp>
        <QueryClientProvider client={client}>
          <BrowserRouter>
            <App />
          </BrowserRouter>
        </QueryClientProvider>
      </AntApp>
    </ConfigProvider>
  </StrictMode>,
);
