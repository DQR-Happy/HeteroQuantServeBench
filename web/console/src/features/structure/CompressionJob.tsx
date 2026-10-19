import { Link } from 'react-router-dom';
import { JsonView, Panel, Status } from '../../components';
import type { StructureRun } from './types';
export default function CompressionJob({ run }: { run: StructureRun }) {
  const result = run.metrics.structure_artifact;
  return (
    <Panel title="独立 CPU 压缩任务">
      <p>
        <Link to={`/runs/${run.id}`}>{run.id}</Link> · <Status state={run.state} /> · 清理：
        {run.cleanup}
      </p>
      <p>
        任务创建独立副本并保存派生权重；不会替换当前驻留模型。构建完成后，需在“制品与加载”显式应用，再执行独立质量评测。
      </p>
      {run.error && <p role="alert">{run.error}</p>}
      {run.progress != null && <JsonView value={run.progress} />}
      {result && (
        <>
          <p>已保存：{result.artifact_id}</p>
          <Link
            to={`/model-structure/artifacts?deployment=${encodeURIComponent(run.config.deployment.id)}&artifact=${result.artifact_id}`}
          >
            查看压缩制品与加载 →
          </Link>
          <details>
            <summary>构建统计与模块变更</summary>
            <JsonView value={result} />
          </details>
        </>
      )}
    </Panel>
  );
}
