#!/usr/bin/env node
// potion-search CLI — Zero-dependency incremental Chinese/English hybrid search over any Markdown directory.
//
// Usage:
//   potion-search "<query>" [--dir <folder>] [--top 8] [--path <substr>] [--json]
//   potion-search build-vocab [--out <vocab_dir>] [--pack <archive.tar.gz>]

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { parseArgs } from "node:util";
import {
  DIM,
  fnv1a,
  getVocabCacheDir,
  buildVocabFromHf,
  packTarGzFlat,
  ensureVocab,
  loadVocab,
  embedText,
  quantizeRow,
  encodeChunkTerms,
  buildBm25FromEncodedTerms,
  writeIndexFiles,
  hasIndexFiles,
  loadIndexFiles,
  queryHybridIndex
} from "../src/index.mjs";

const RRF_K = 60;

function walkMarkdownFiles(rootDir, out = []) {
  for (const ent of fs.readdirSync(rootDir, { withFileTypes: true })) {
    if (ent.name.startsWith(".") || ent.name === "node_modules") continue;
    const full = path.join(rootDir, ent.name);
    if (ent.isDirectory()) {
      walkMarkdownFiles(full, out);
    } else if (ent.isFile() && ent.name.endsWith(".md")) {
      const st = fs.statSync(full);
      out.push({
        relPath: path.relative(rootDir, full).replace(/\\/g, "/"),
        fullPath: full,
        stamp: `${Math.floor(st.mtimeMs)}:${st.size}`
      });
    }
  }
  return out;
}

async function syncDirectoryIndex(targetDir, indexDir) {
  const t0 = performance.now();
  const absDir = path.resolve(targetDir);
  const files = walkMarkdownFiles(absDir).sort((a, b) => a.relPath.localeCompare(b.relPath));

  let prevLoaded = null;
  const prevChunksByFile = new Map();
  if (hasIndexFiles(indexDir) && fs.existsSync(path.join(indexDir, "doc-terms.txt"))) {
    try {
      const metaOnly = JSON.parse(fs.readFileSync(path.join(indexDir, "meta.json"), "utf8"));
      const prevStamps = metaOnly.fileStamps || {};
      let anyChanged = Object.keys(prevStamps).length !== files.length;
      if (!anyChanged) {
        for (const f of files) {
          if (prevStamps[f.relPath] !== f.stamp) {
            anyChanged = true;
            break;
          }
        }
      }
      if (!anyChanged) return;

      prevLoaded = loadIndexFiles(indexDir, { lazyChunks: false, loadDocTerms: true });
      if (prevLoaded.docTerms && prevLoaded.docTerms.length === prevLoaded.numChunks) {
        for (let i = 0; i < prevLoaded.numChunks; i++) {
          const c = prevLoaded.chunks[i];
          let list = prevChunksByFile.get(c.p);
          if (!list) {
            list = [];
            prevChunksByFile.set(c.p, list);
          }
          list.push(i);
        }
      } else {
        prevLoaded = null;
      }
    } catch {
      prevLoaded = null;
    }
  }

  const prevStamps = prevLoaded?.meta?.fileStamps || {};
  const nextStamps = {};
  const changedFiles = [];

  for (const f of files) {
    nextStamps[f.relPath] = f.stamp;
    if (!prevLoaded || prevStamps[f.relPath] !== f.stamp) {
      changedFiles.push(f);
    }
  }

  const vocabDir = await ensureVocab();
  const vocab = loadVocab(vocabDir, { preloadVectors: changedFiles.length > 10 });
  const scratch = new Float32Array(DIM);
  const newFileData = new Map();

  for (const f of changedFiles) {
    const content = fs.readFileSync(f.fullPath, "utf8");
    const fileTitle = path.basename(f.relPath, ".md");
    const sections = content.split(/\n(?=#{1,3}\s)/);
    const fileChunks = [];

    for (const sec of sections) {
      const trimmed = sec.trim();
      if (trimmed.length < 25) continue;
      const flEnd = trimmed.indexOf("\n");
      const firstLine = flEnd === -1 ? trimmed : trimmed.slice(0, flEnd);
      const heading = firstLine.startsWith("#") ? firstLine.slice(0, 120) : fileTitle;
      const snippet = trimmed.replace(/\s+/g, " ").slice(0, 220);
      const combined = `${f.relPath.replace(/[/_-]/g, " ")}\n${heading}\n${trimmed.slice(0, 1400)}`;

      const rowI8 = new Int8Array(DIM);
      const rowScale = new Float32Array(1);
      embedText(combined, vocab, scratch);
      quantizeRow(scratch, rowI8, 0, rowScale);

      fileChunks.push({
        chunk: { p: f.relPath, h: heading, s: snippet },
        rowI8,
        scale: rowScale[0],
        encodedTerms: encodeChunkTerms(combined)
      });
    }
    newFileData.set(f.relPath, fileChunks);
  }

  const finalChunks = [];
  const finalDocTerms = [];
  const rowSources = [];

  for (const f of files) {
    const updated = newFileData.get(f.relPath);
    if (updated !== undefined) {
      for (const item of updated) {
        finalChunks.push(item.chunk);
        finalDocTerms.push(item.encodedTerms);
        rowSources.push({ kind: "new", rowI8: item.rowI8, scale: item.scale });
      }
    } else {
      const oldIndices = prevChunksByFile.get(f.relPath) || [];
      for (const oldIdx of oldIndices) {
        finalChunks.push(prevLoaded.chunks[oldIdx]);
        finalDocTerms.push(prevLoaded.docTerms[oldIdx]);
        rowSources.push({ kind: "old", oldIdx });
      }
    }
  }

  const numChunks = finalChunks.length;
  const vecI8 = new Int8Array(numChunks * DIM);
  const vecScales = new Float32Array(numChunks);

  for (let i = 0; i < numChunks; i++) {
    const src = rowSources[i];
    if (src.kind === "new") {
      vecI8.set(src.rowI8, i * DIM);
      vecScales[i] = src.scale;
    } else {
      const oldBase = src.oldIdx * DIM;
      vecI8.set(prevLoaded.vecI8.subarray(oldBase, oldBase + DIM), i * DIM);
      vecScales[i] = prevLoaded.vecScales[src.oldIdx];
    }
  }

  const bm25 = buildBm25FromEncodedTerms(finalDocTerms);
  writeIndexFiles(
    indexDir,
    {
      dir: absDir,
      updatedAt: new Date().toISOString(),
      fileCount: files.length,
      chunkCount: numChunks,
      fileStamps: nextStamps,
      chunks: finalChunks
    },
    vecI8,
    vecScales,
    bm25,
    finalDocTerms
  );

  const elapsedMs = Math.round(performance.now() - t0);
  process.stderr.write(
    `[potion-search] Synced ${numChunks} chunks (${changedFiles.length}/${files.length} files updated) in ${elapsedMs}ms\n`
  );
}

async function main() {
  const { values, positionals } = parseArgs({
    options: {
      dir: { type: "string", short: "d", default: "." },
      top: { type: "string", short: "k", default: "8" },
      path: { type: "string", short: "p" },
      out: { type: "string" },
      pack: { type: "string" },
      json: { type: "boolean", default: false }
    },
    allowPositionals: true
  });

  if (positionals[0] === "build-vocab") {
    const outDir = values.out ? path.resolve(values.out) : getVocabCacheDir();
    await buildVocabFromHf(outDir);
    if (values.pack) {
      packTarGzFlat(
        outDir,
        ["tokens.bin", "hash.bin", "scores.f32", "vectors.i8", "scales.f32"],
        path.resolve(values.pack)
      );
      process.stderr.write(`[potion-search] Packed archive -> ${path.resolve(values.pack)}\n`);
    }
    return;
  }

  const query = positionals[0] === "search" ? positionals.slice(1).join(" ") : positionals.join(" ");
  if (!query.trim()) {
    process.stderr.write(
      "Usage:\n" +
        '  potion-search "<query>" [--dir <folder>] [--top 8] [--path <substr>] [--json]\n' +
        "  potion-search build-vocab [--out <vocab_dir>] [--pack <archive.tar.gz>]\n"
    );
    process.exit(1);
  }

  const targetDir = path.resolve(values.dir);
  const dirKey = fnv1a(targetDir).toString(16).padStart(8, "0");
  const indexDir = path.join(os.homedir(), ".cache", "potion-search", `dir-${dirKey}`);

  await syncDirectoryIndex(targetDir, indexDir);

  const vocabDir = await ensureVocab();
  const vocab = loadVocab(vocabDir, { preloadVectors: false });
  const index = loadIndexFiles(indexDir, { lazyChunks: true });

  const pathFilter = values.path ? values.path.toLowerCase() : null;
  const filterFn = pathFilter ? (c) => c.p.toLowerCase().includes(pathFilter) : null;

  const { vecRanked, bm25Ranked, vecScores } = queryHybridIndex(query, index, vocab, { filterFn });
  const { getChunk } = index;

  const rrfScores = new Map();
  const bestByFile = new Map();

  const seenVec = new Set();
  let vecRank = 0;
  for (const item of vecRanked) {
    const c = getChunk(item.idx);
    if (seenVec.has(c.p)) continue;
    seenVec.add(c.p);
    vecRank++;
    rrfScores.set(c.p, (rrfScores.get(c.p) ?? 0) + 1 / (RRF_K + vecRank));
    bestByFile.set(c.p, { ...c, vecSimilarity: Number(item.score.toFixed(4)), bm25Score: null });
  }

  const seenBm25 = new Set();
  let bm25Rank = 0;
  for (const item of bm25Ranked) {
    const c = getChunk(item.idx);
    if (seenBm25.has(c.p)) continue;
    seenBm25.add(c.p);
    bm25Rank++;
    rrfScores.set(c.p, (rrfScores.get(c.p) ?? 0) + 1 / (RRF_K + bm25Rank));
    const entry = bestByFile.get(c.p) ?? {
      ...c,
      vecSimilarity: Number(vecScores[item.idx].toFixed(4))
    };
    entry.bm25Score = Number(item.score.toFixed(2));
    if (bm25Rank <= 10) {
      entry.h = c.h;
      entry.s = c.s;
    }
    bestByFile.set(c.p, entry);
  }

  const topK = Math.max(1, Number(values.top) || 8);
  const results = [...rrfScores.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, topK)
    .map(([filePath, score]) => {
      const info = bestByFile.get(filePath);
      return {
        path: filePath,
        heading: info.h,
        snippet: info.s,
        rrfScore: Number(score.toFixed(4)),
        vecSimilarity: info.vecSimilarity,
        bm25Score: info.bm25Score
      };
    });

  if (values.json) {
    console.log(JSON.stringify({ dir: targetDir, chunkCount: index.numChunks, results }, null, 2));
    return;
  }

  console.log(`${targetDir} (${index.numChunks} chunks)`);
  for (let i = 0; i < results.length; i++) {
    const r = results[i];
    console.log(
      `\n[${i + 1}] ${r.path}  (rrf=${r.rrfScore}, vec=${r.vecSimilarity ?? "-"}, bm25=${r.bm25Score ?? "-"})`
    );
    console.log(`    ${r.heading}`);
    console.log(`    ${r.snippet}`);
  }
}

main().catch((err) => {
  process.stderr.write(`[potion-search] Error: ${err.message}\n`);
  process.exit(1);
});
