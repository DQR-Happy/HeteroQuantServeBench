import fs from 'node:fs';
import path from 'node:path';
import ts from 'typescript';

function files(dir) {
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .flatMap((e) =>
      e.isDirectory()
        ? files(path.join(dir, e.name))
        : e.name.endsWith('.tsx')
          ? [path.join(dir, e.name)]
          : [],
    );
}
const inventory = [];
for (const file of files('src')) {
  if (file.endsWith('/ActionButton.tsx')) continue;
  const source = fs.readFileSync(file, 'utf8');
  const ast = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  function walk(node) {
    if (
      (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node)) &&
      node.tagName.getText(ast) === 'Button'
    ) {
      const attrs = node.attributes.properties.filter(ts.isJsxAttribute);
      if (attrs.some((a) => a.name.getText(ast) === 'disabled')) {
        const disabled = attrs.find((a) => a.name.getText(ast) === 'disabled');
        const parent = ts.isJsxOpeningElement(node) ? node.parent : node;
        inventory.push({
          file,
          line: ast.getLineAndCharacterOfPosition(node.getStart(ast)).line + 1,
          start: node.getStart(ast),
          end: node.end,
          expression: disabled.initializer?.getText(ast),
          label: parent.getText(ast).slice(-250),
          explained:
            source.includes('components/ActionButton') &&
            attrs.some((a) => a.name.getText(ast) === 'disabledReason'),
        });
      }
    }
    ts.forEachChild(node, walk);
  }
  walk(ast);
}
if (process.argv.includes('--list')) console.log(JSON.stringify(inventory, null, 2));
else {
  const missing = inventory.filter((n) => !n.explained);
  if (missing.length) {
    console.error(
      missing.map((n) => `${n.file}:${n.line} disabled action needs disabledReason`).join('\n'),
    );
    process.exitCode = 1;
  } else
    console.log(
      `Disabled action contract: ${inventory.length} controls have explicit explanations.`,
    );
}
