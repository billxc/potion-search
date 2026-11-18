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
  queryHybridIndex
} from "../src/index.mjs";

test("hybrid search ranks matching Chinese and English chunks", async () => {
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
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});
