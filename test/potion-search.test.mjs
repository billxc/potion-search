import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  DIM,
  ensureVocab,
  loadVocab,
  embedText,
  quantizeRow,
  encodeChunkTerms,
  buildBm25FromEncodedTerms,
  writeIndexFiles,
  loadIndexFiles,
  queryHybridIndex,
  syncHybridIndex,
  packIndexBundle,
   extractTarGzFlat,
  openHybridIndex
} from "../src/index.mjs";
import { walkMarkdownFiles } from "../bin/potion-search.mjs";

test("hybrid search, BM25 binary hash table, OOV guard, and close()", async () => {
  const vocabDir = await ensureVocab();
  const vocab = loadVocab(vocabDir, { preloadVectors: false });

  const docs = [
    {
      p: "docs/webview2-mac.md",
      h: "# WebView2 Mac Crash Dump 分析",
      s: "使用 lldb 和 minidump_dump 分析 macOS 上的崩溃日志与符号表"
    },
    {
      p: "docs/adguard-dns.md",
      h: "# AdGuard Home 日本机 DoH 配置",
      s: "在 xray-jp 上通过 Caddy 配置多 IP 裸 IP TLS 证书与 DNS 分流"
    }
  ];

  const vecI8 = new Int8Array(docs.length * DIM);
  const vecScales = new Float32Array(docs.length);
  const scratch = new Float32Array(DIM);
  const termLines = [];

  for (let i = 0; i < docs.length; i++) {
    const txt = `${docs[i].p}\n${docs[i].h}\n${docs[i].s}`;
    embedText(txt, vocab, scratch);
    quantizeRow(scratch, vecI8, i, vecScales);
    termLines.push(encodeChunkTerms(txt));
  }

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "potion-search-test-"));
  try {
    const bm25 = buildBm25FromEncodedTerms(termLines);
    writeIndexFiles(tmpDir, { chunks: docs }, vecI8, vecScales, bm25, termLines);

    const index = loadIndexFiles(tmpDir, { lazyChunks: true });
    const res1 = queryHybridIndex("WebView2 崩溃调试 lldb", index, vocab);
    assert.equal(res1.vecRanked[0].idx, 0);
    assert.equal(res1.bm25Ranked[0].idx, 0);

    const res2 = queryHybridIndex("日本 DNS DoH 证书", index, vocab);
    assert.equal(res2.vecRanked[0].idx, 1);
    assert.equal(res2.bm25Ranked[0].idx, 1);

    const oovRes = queryHybridIndex("~~~ $$$ ^^^", index, vocab);
    assert.equal(oovRes.vecRanked.length, 0);

    const cppVec = embedText("c++ utf-8 node.js", vocab, new Float32Array(DIM));
    assert.ok(cppVec.some((v) => v !== 0));

    index.close();
    index.close();
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test("walkMarkdownFiles preserves nested relative paths, syncHybridIndex two-stage incremental, packIndexBundle, and openHybridIndex", async () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "potion-store-test-"));
  const docsDir = path.join(tmpRoot, "docs");
  const idxDir = path.join(tmpRoot, "index");
  const unpackDir = path.join(tmpRoot, "unpacked");
  const bundlePath = path.join(tmpRoot, "bundle.tar.gz");

  try {
    fs.mkdirSync(path.join(docsDir, "sub1"), { recursive: true });
    fs.mkdirSync(path.join(docsDir, "sub2"), { recursive: true });
    fs.writeFileSync(
      path.join(docsDir, "sub1", "Overview.md"),
      "# WebView2 Mac Overview\nUsing lldb to debug WebView2 Mac crash dump and Mojo IPC."
    );
    fs.writeFileSync(
      path.join(docsDir, "sub2", "Overview.md"),
      "# Deprecated Legacy Guide\nOld deprecated instructions for legacy proxy setup."
    );

    const walked = walkMarkdownFiles(docsDir).map((f) => f.sourceId).sort();
    assert.deepEqual(walked, ["sub1/Overview.md", "sub2/Overview.md"]);

    const loadedCalls = [];
    const sourcesState = [
      { sourceId: "sub1/Overview.md", stamp: "v1" },
      { sourceId: "sub2/Overview.md", stamp: "v1" }
    ];

    const r1 = await syncHybridIndex(idxDir, {
      listSources: () => sourcesState,
      loadChunks: (changed) => {
        loadedCalls.push(changed.map((c) => c.sourceId));
        return changed.map((c) => ({
          sourceId: c.sourceId,
          groupKey: c.sourceId,
          h: c.sourceId,
          s: fs.readFileSync(path.join(docsDir, c.sourceId), "utf8"),
          text: fs.readFileSync(path.join(docsDir, c.sourceId), "utf8")
        }));
      }
    });
    assert.equal(r1.updatedSources, 2);
    assert.equal(loadedCalls.length, 1);

    // Unchanged run -> 0 loadChunks calls
    const r2 = await syncHybridIndex(idxDir, {
      listSources: () => sourcesState,
      loadChunks: () => {
        throw new Error("Should not be called when stamps match");
      }
    });
    assert.equal(r2.updatedSources, 0);

    // Pack read-only bundle -> strips doc-terms.txt and fileStamps
    packIndexBundle(idxDir, bundlePath);
    extractTarGzFlat(bundlePath, unpackDir);
    assert.equal(fs.existsSync(path.join(unpackDir, "doc-terms.txt")), false);
    const bundledMeta = JSON.parse(fs.readFileSync(path.join(unpackDir, "meta.json"), "utf8"));
    assert.equal(bundledMeta.fileStamps, undefined);

    const store = await openHybridIndex(unpackDir);
    const hits = store.search("Overview", {
      topK: 2,
      adjustScore: (gk, _c, s) => (gk.includes("sub2") ? s * 0.1 : s)
    });
    assert.equal(hits[0].path, "sub1/Overview.md");
    store.close();
  } finally {
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  }
});
