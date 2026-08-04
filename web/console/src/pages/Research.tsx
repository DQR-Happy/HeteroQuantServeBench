import { useQuery } from '@tanstack/react-query';
import { Alert, Button, Space, Tabs, Tag } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import { api, base } from '../api/client';
import { PageHead, QueryState } from '../components';
import CapabilityMatrix from '../features/observability/Capabilities';
import PhaseExplorer from '../features/research/PhaseExplorer';
import QuantizationEvidence from '../features/research/QuantizationEvidence';
import type { Research } from '../features/research/types';

export default function ResearchPage() {
  const query = useQuery({
    queryKey: ['research'],
    queryFn: () => api<Research>('/research'),
    staleTime: 60000,
  });
  return (
    <>
      <PageHead
        kicker="OBSERVE · ATTRIBUTE · VERIFY"
        title="深度分析"
        description="沿着阶段、框架算子、GPU kernel 和硬件计数器追溯已有实验；把量化的存储收益与真实执行分开核验。"
        extra={
          <Space wrap>
            <Button href={`${base}/research/export`}>导出研究索引</Button>
            <Button
              icon={<ReloadOutlined />}
              loading={query.isFetching}
              onClick={() => void query.refetch()}
            >
              刷新证据
            </Button>
          </Space>
        }
      />
      <Alert
        type="info"
        showIcon
        className="section-gap"
        message={
          <>
            <Tag color="blue">历史采集</Tag> 当前页面来自阶段实验的真实制品
          </>
        }
        description="历史采集不代表正在运行的模型。查看本次请求请进入运行详情的「全链路观测」；新的采集必须在推理提交前开启。"
      />
      <QueryState loading={query.isPending} error={query.error} />
      {query.data && (
        <Tabs
          items={[
            {
              key: 'phases',
              label: '阶段与内核钻取',
              children: <PhaseExplorer profiling={query.data.profiling} />,
            },
            {
              key: 'quantization',
              label: '量化执行真实性',
              children: <QuantizationEvidence quantization={query.data.quantization} />,
            },
            { key: 'capabilities', label: '当前采集能力', children: <CapabilityMatrix /> },
          ]}
        />
      )}
    </>
  );
}
