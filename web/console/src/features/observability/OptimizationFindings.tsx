import { useQuery } from '@tanstack/react-query';
import { Alert, Empty, Tag } from 'antd';
import { api } from '../../api/client';
import { JsonView, QueryState } from '../../components';

type Finding = {
  id: string;
  title: string;
  evidence: unknown;
  action: string;
  validation: string;
  priority: string;
};
type Optimization = {
  status: string;
  findings: Finding[];
  constraints: string[];
  measurement_profile: string;
  trace: unknown;
};

export default function OptimizationFindings({ runId }: { runId: string }) {
  const query = useQuery({
    queryKey: ['optimization', runId],
    queryFn: () => api<Optimization>(`/runs/${runId}/optimization`),
    staleTime: 60000,
  });
  return (
    <>
      <Alert
        type="warning"
        showIcon
        className="section-gap"
        message="以下是证据驱动的候选假设，尚未证明优化有效。"
        description="先复现现象，再隔离变量；使用未插桩基准验证速度、质量与内存，并保留失败结果。可在上方导出完整优化方案。"
      />
      <QueryState loading={query.isPending} error={query.error} />
      {query.data?.findings.map((finding) => (
        <article className="finding-card" key={finding.id}>
          <h3>
            <Tag>{finding.priority}</Tag>
            {finding.title}
          </h3>
          <div className="finding-evidence">
            <strong>当前证据</strong>
            {typeof finding.evidence === 'string' ? (
              <p>{finding.evidence}</p>
            ) : (
              <JsonView value={finding.evidence} />
            )}
          </div>
          <p>
            <strong>候选操作：</strong>
            {finding.action}
          </p>
          <p>
            <strong>验证方法：</strong>
            {finding.validation}
          </p>
        </article>
      ))}
      {query.data && !query.data.findings.length && (
        <Empty description="当前证据尚不足以形成具体优化假设" />
      )}
      {query.data && (
        <div className="help-text">
          {query.data.constraints.map((constraint, index) => (
            <p key={index}>{constraint}</p>
          ))}
        </div>
      )}
    </>
  );
}
