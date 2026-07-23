import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const templateRoot = new URL("../", import.meta.url);

async function render(pathname = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`http://localhost${pathname}`, {
      headers: { accept: "text/html" },
    }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the Shallow Red game shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Shallow Red/);
  assert.match(html, /You can(?:&apos;|&#x27;|')t/);
  assert.doesNotMatch(html, /Your challenge is stranger/);
  assert.match(html, /Try to lose\./);
  assert.match(html, /Choose your color/);
  assert.match(html, />Black</);
  assert.match(html, /as quickly as possible/);
  assert.match(html, /href="\/technical"/);
  assert.match(html, /Shallow Red(?:&apos;|&#x27;|')s record/);
  assert.match(html, />Losses</);
  assert.match(html, />Wins</);
  assert.match(html, />Draws</);
  assert.match(html, />Total games</);
  assert.match(html, /Loading tiny neural model/);
  assert.doesNotMatch(html, /Shallow Red losses|Shallow Red wins/i);
  assert.doesNotMatch(html, /Accidental AI wins|Last search|research evaluations|How this scales|server bill|moves computed locally/i);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Starter Project/);
});

test("server-renders the technical page and model credit", async () => {
  const response = await render("/technical");
  assert.equal(response.status, 200);

  const html = await response.text();
  assert.match(html, /The Technical Stuff/);
  assert.match(html, /By <strong>Jian Ouyang<\/strong>/);
  assert.match(html, /What does .*losing.* mean/);
  assert.match(html, /281 of 300/);
  assert.match(html, /39\.7 KB/);
  assert.match(html, /shortlists twelve legal moves/);
  assert.match(html, /gpt-5\.6-sol-high/);
  assert.match(html, /Back to the game/);
});

test("removes the disposable starter preview and fixes board rows", async () => {
  const [page, layout, packageJson, styles] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  assert.match(page, /<ShallowRedGame \/>/);
  assert.doesNotMatch(page, /You can&apos;t<br/);
  assert.match(layout, /Shallow Red — You Can't Lose/);
  assert.match(packageJson, /"chess\.js"/);
  assert.match(styles, /grid-template-rows: repeat\(8, minmax\(0, 1fr\)\)/);
  assert.match(styles, /white-space: nowrap/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  await assert.rejects(access(new URL("../app/_sites-preview", templateRoot)));
});
