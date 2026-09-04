import {
  ExperimentOutlined,
  FileSearchOutlined,
  HomeOutlined,
  NodeIndexOutlined,
  PlayCircleOutlined,
  ClusterOutlined,
} from '@ant-design/icons';

export const navigation = [
  {
    id: 'workspace',
    label: '工作空间',
    icon: <HomeOutlined />,
    path: '/',
    pages: [
      { path: '/', label: '项目概览', detail: '近期运行与执行环境' },
      { path: '/showcase', label: '技术导览', detail: '沿真实证据介绍工程决策' },
    ],
  },
  {
    id: 'benchmarks',
    label: '评测',
    icon: <ExperimentOutlined />,
    path: '/benchmarks',
    pages: [
      {
        path: '/benchmarks',
        label: '性能与质量对照',
        detail: '固定典型输入，多方案输出与性能比较',
      },
    ],
  },
  {
    id: 'runs',
    label: '运行',
    icon: <PlayCircleOutlined />,
    path: '/runs',
    pages: [
      { path: '/runs', label: '运行记录', detail: '检索任务、状态和冻结配置' },
      { path: '/playground', label: '推理工作台', detail: '交互推理案例、具体实现与观测' },
    ],
  },
  {
    id: 'observe',
    label: '观测',
    icon: <NodeIndexOutlined />,
    path: '/observations',
    pages: [
      { path: '/observations', label: '机制观测', detail: '内存、KV、分配器与 CPU 页证据' },
      { path: '/memory-flow', label: '内存与数据流', detail: '空间结构、Tensor 与数据传输' },
      { path: '/research', label: '深度分析', detail: '阶段、算子、GPU kernel 与计数器' },
    ],
  },
  {
    id: 'optimize',
    label: '优化',
    icon: <ExperimentOutlined />,
    path: '/optimization',
    pages: [
      { path: '/optimization', label: '优化实验', detail: '冻结计划、执行凭据与质量门' },
      { path: '/compare', label: '运行对比', detail: '基线 A 与候选 B 的受控对照' },
      { path: '/quantization', label: '量化与质量', detail: '制品、实际执行路径和质量证据' },
      { path: '/efficiency', label: '性能与能效', detail: '历史性能和能量计量范围' },
    ],
  },
  {
    id: 'evidence',
    label: '证据',
    icon: <FileSearchOutlined />,
    path: '/evidence',
    pages: [
      { path: '/evidence', label: '证据中心', detail: '归档判定、报告与原始文件' },
      { path: '/experiments', label: '实验地图', detail: '阶段实验及其验收结果' },
      { path: '/kernels', label: '算子与配置', detail: '源码、实现与配置目录' },
    ],
  },
  {
    id: 'environment',
    label: '环境',
    icon: <ClusterOutlined />,
    path: '/devices',
    pages: [
      { path: '/devices', label: '设备与部署', detail: '节点资源与模型生命周期' },
      { path: '/settings', label: '设置与帮助', detail: '会话、能力与指标契约' },
    ],
  },
];
export const destinations = navigation.flatMap((group) =>
  group.pages.map((page) => ({ ...page, group: group.label, icon: group.icon })),
);
