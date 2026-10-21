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
      {
        path: '/scheduling',
        label: '多请求调度实验',
        detail: 'L8 连续合批、KV 生命周期与质量对照',
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
      { path: '/diagnostics', label: '执行诊断', detail: '原生 nsys 时间线与 ncu 硬件计数器' },
    ],
  },
  {
    id: 'optimize',
    label: '优化',
    icon: <ExperimentOutlined />,
    path: '/optimization',
    pages: [
      { path: '/optimization', label: '优化实验', detail: '冻结计划、执行凭据与质量门' },
      { path: '/engines', label: '原生引擎调优', detail: 'vLLM / SGLang 分层参数、生命周期与公共压测' },
      { path: '/system', label: '硬件与系统配置', detail: 'L1 CPU/NUMA、线程、GPU 功耗与配置对照' },
      {
        path: '/application',
        label: '服务与应用优化',
        detail: 'L10 路由、结果缓存、Token 预算与托管副本弹性',
      },
      {
        path: '/model-structure',
        label: '模型结构与压缩',
        detail: 'L5 剪枝、真实缩维、低秩制品与质量对照',
      },
      {
        path: '/memory-cache',
        label: '内存与 KV 管理',
        detail: 'L6 动态/静态缓存、卸载预取、重计算与前缀复用',
      },
      {
        path: '/decoding',
        label: '解码与生成算法',
        detail: 'L7 采样、推测验证、独立草稿与约束解码',
      },
      {
        path: '/parallel',
        label: '多卡与分布式执行',
        detail: 'L9 DP、TP、按层放置与 Prefill/Decode KV 传输',
      },
      {
        path: '/precision',
        label: '表示精度与量化',
        detail: 'L4 主流方法、在线执行、制品与独立质量评测',
      },
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
      { path: '/model-architecture', label: '模型架构', detail: '制品模块、层结构与逻辑计算依赖' },
      { path: '/models', label: '模型下载', detail: 'ModelScope 发布方权重、版本与下载进度' },
      {
        path: '/models/local',
        label: '本地模型管理',
        detail: '模型目录、磁盘空间、部署引用与清理',
      },
      { path: '/machines', label: '机器连接', detail: 'SSH / HTTP 节点管理与切换' },
      { path: '/guide', label: '使用指南', detail: '逐页操作、替换范围与证据解读' },
      { path: '/settings', label: '设置与帮助', detail: '会话、能力与指标契约' },
    ],
  },
];
export const destinations = navigation.flatMap((group) =>
  group.pages.map((page) => ({ ...page, group: group.label, icon: group.icon })),
);
