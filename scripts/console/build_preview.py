"""Build an offline, read-only screenshot gallery; no models or hardware imports."""

import argparse
import base64
import html
from pathlib import Path


PAGES = (
    ("workspace", "工作空间", "workspace-desktop.png"),
    ("runs", "运行目录", "runs-desktop.png"),
    ("playground", "推理配置", "playground-desktop.png"),
    ("optimization", "优化实验", "optimization-connected.png"),
    ("kv", "KV 布局", "kv-desktop.png"),
    ("allocator", "分配与回收", "allocator-desktop.png"),
    ("pages", "CPU 页样本", "pages-desktop.png"),
    ("links", "算子关联", "correlation-desktop.png"),
    ("compare", "A/B 对照", "compare-desktop.png"),
    ("mobile", "手机界面", "compare-mobile.png"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshots", type=Path, default=Path("reports/console/v04"))
    parser.add_argument("--output", type=Path, default=Path("web/console/public/preview.html"))
    args = parser.parse_args()
    navigation, sections = [], []
    for index, (key, title, name) in enumerate(PAGES):
        path = args.screenshots / name
        raw = path.read_bytes()
        if not raw.startswith(b"\x89PNG\r\n\x1a\n") or len(raw) > 8 * 1024**2:
            raise ValueError(f"Invalid or oversized screenshot: {name}")
        encoded = base64.b64encode(raw).decode("ascii")
        navigation.append(f'<a href="#{key}" data-page="{key}">{html.escape(title)}</a>')
        sections.append(
            f'<section id="{key}" aria-labelledby="title-{key}"{ " hidden" if index else "" }>'
            f'<h2 id="title-{key}">{html.escape(title)}</h2>'
            '<p class="caption">真实归档数据的界面截图 · 图片中的按钮不可操作</p>'
            f'<img src="data:image/png;base64,{encoded}" alt="HQSB {html.escape(title)}界面"'
            f'{ " class=mobile" if key == "mobile" else "" }></section>'
        )
    template = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HQSB · 离线界面预览</title><style>
*{box-sizing:border-box}body{margin:0;background:#f6f7f8;color:#20262e;font:14px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif}
header{padding:20px 28px;background:#fff;border-bottom:1px solid #d8dce1}h1{font-size:22px;margin:0 0 6px}header p{margin:0;color:#586372}
.layout{display:grid;grid-template-columns:180px minmax(0,1fr)}nav{padding:20px 12px;display:flex;flex-direction:column;gap:6px;align-self:start;position:sticky;top:0}
nav a{padding:9px 12px;color:#435f82;text-decoration:none;border-radius:6px}nav a[aria-current=page]{background:#dfe8f2;color:#304e72}nav a:hover{background:#e9ecef}
main{padding:24px;min-width:0}h2{font-size:18px;margin:0 0 6px}.caption{color:#586372;margin:0 0 16px}img{display:block;width:100%;height:auto;border:1px solid #d8dce1;border-radius:6px}img.mobile{max-width:390px}
a:focus-visible{outline:2px solid #435f82;outline-offset:2px}footer{padding:16px 28px;color:#586372;font-size:12px}
@media(max-width:700px){.layout{grid-template-columns:minmax(0,1fr)}nav{position:static;flex-direction:row;flex-wrap:wrap;padding:12px}main{padding:12px}header{padding:16px}}
</style></head><body><header><h1>HQSB 离线界面预览</h1><p>UI v0.4 · 全新项目工作台 · 真实界面截图 · 不连接后端、不执行推理。左侧可切换页面；实际操作请访问已启动的工作台。</p></header>
<div class="layout"><nav aria-label="预览页面">NAVIGATION</nav><main>SECTIONS</main></div>
<footer>图片已嵌入此 HTML，单个文件即可离线打开。采集完整仅指声明范围，不代表所有物理内存和硬件访问可见。</footer>
<script>
function selectPage(){
  const key=location.hash.slice(1)||'workspace';
  const target=document.getElementById(key);
  if(!target||target.tagName!=='SECTION'){location.hash='workspace';return;}
  document.querySelectorAll('main section').forEach(section=>section.hidden=section!==target);
  document.querySelectorAll('nav a').forEach(link=>{
    if(link.dataset.page===key)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');
  });
}
addEventListener('hashchange',selectPage);selectPage();
</script></body></html>"""
    output = template.replace("NAVIGATION", "".join(navigation)).replace("SECTIONS", "".join(sections))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"Offline screenshot preview: {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
