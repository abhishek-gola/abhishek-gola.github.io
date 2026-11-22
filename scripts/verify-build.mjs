import assert from 'node:assert/strict';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

function files(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry =>
    entry.isDirectory() ? files(join(directory, entry.name)) : [join(directory, entry.name)]);
}
const pages = files('dist').filter(path => path.endsWith('.html'));
assert(pages.length > 0, 'Build the site before verification.');
const documents = files('dist').filter(path => /\.(html|xml)$/.test(path));
const published = documents.map(path => readFileSync(path, 'utf8')).join('\n');
for (const path of files('src/content/writing').filter(path => /\.mdx?$/.test(path))) {
  const frontmatter = readFileSync(path, 'utf8').split('---')[1] || '';
  if (/^draft:\s*true\s*$/m.test(frontmatter)) {
    const slug = path.replace('src/content/writing/', '').replace(/\.mdx?$/, '');
    assert(!existsSync(`dist/writing/${slug}/index.html`), `Draft page generated: ${slug}`);
    assert(!published.includes(`/writing/${slug}/`), `Draft linked from published output: ${slug}`);
  }
}
for (const page of pages) {
  const html = readFileSync(page, 'utf8');
  assert(html.includes('rel="canonical"'), `${page} lacks a canonical URL`);
  for (const [, url] of html.matchAll(/(?:href|src)="(\/[^"#?]*)/g)) {
    if (url.startsWith('//')) continue;
    const path = decodeURIComponent(url).replace(/^\//, '');
    assert(existsSync(join('dist', path)) || existsSync(join('dist', path, 'index.html')), `${page}: broken local link ${url}`);
  }
}
const home = readFileSync('dist/index.html', 'utf8');
assert(home.indexOf('Engineering, explained.') < home.indexOf('Ideas in writing.'), 'Speaking evidence must precede writing.');
for (const id of ['YTMqfx-a9cw','LVAQyazpbu8','me4ylB8OsD4','O0oqOvsZZ8o']) assert(home.includes(id), `Missing recording ${id}`);
assert(!home.includes('tensor-visual'), 'Decorative hero should not return to the publication homepage.');
const rss = readFileSync('dist/rss.xml', 'utf8');
const dates = [...rss.matchAll(/<pubDate>(.*?)<\/pubDate>/g)].map(([, value]) => Date.parse(value));
assert(dates.every((date, index) => index === 0 || dates[index - 1] >= date), 'RSS must be newest first.');
console.log(`Verified ${pages.length} pages: draft exclusion, local links, canonical URLs, speaking priority, recording links, and RSS ordering.`);
