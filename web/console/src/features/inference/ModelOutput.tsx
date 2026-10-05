import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import './model-output.css';

const plugins = [remarkGfm, remarkMath];
const components: Components = {
  table: ({ children }) => (
    <div className="model-output-table" role="region" aria-label="输出表格" tabIndex={0}>
      <table>{children}</table>
    </div>
  ),
};

/** Render each accumulated snapshot, including unfinished streaming Markdown. */
export default function ModelOutput({ text, active }: { text: string; active: boolean }) {
  return (
    <div className="model-output output-text" aria-label="模型输出内容" aria-busy={active}>
      <ReactMarkdown
        skipHtml
        remarkPlugins={plugins}
        rehypePlugins={[[rehypeKatex, { trust: false }]]}
        components={components}
      >
        {text}
      </ReactMarkdown>
      {active && <span className="cursor" aria-hidden="true" />}
    </div>
  );
}
