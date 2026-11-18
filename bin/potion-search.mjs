#!/usr/bin/env node
// potion-search CLI — Zero-dependency incremental Chinese/English hybrid search over any Markdown directory.
//
// Usage:
//   potion-search "<query>" [--dir <folder>] [--index <index_dir>] [--top 8] [--path <substr>] [--json]
//   potion-search sync --dir <folder> [--index <index_dir>] [--pack <bundle.tar.gz>]
//   potion-search build-vocab [--out <vocab_dir>] [--pack <archive.tar.gz>]

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";
import {
  fnv1a,
  getVocabCacheDir,
  buildVocabFromHf,
  packTarGzFlat,
  syncHybridIndex,
  packIndexBundle,
  openHybridIndex
} from "../src/index.mjs";

export function walkMarkdownFiles(rootDir, curDir = rootDir, out = []) {
  for (const ent of fs.readdirSync(curDir, { withFileTypes: true })) {
    if (ent.name.startsWith(".") || ent.name === "node_modules") continue;
    const full = path.join(curDir, ent.name);
    if (ent.isDirectory()) {
      walkMarkdownFiles(rootDir, full, out);
    } else if (ent.isFile() && ent.name.endsWith(".md")) {
      const st = fs.statSync(full);
      out.push({
        sourceId: path.relative(rootDir, full).replace(/\\/g, "/"),
        fullPath: full,
        stamp: `${Math.floor(st.mtimeMs)}:${st.size}`
      });
    }
  }
  return out;
}

export function splitMeaningfulSections(content, fileTitle) {
  const rawSecs = content.split(/\n(?=#{1,3}\s)/);
  const merged = [];
  let pending = "";
  for (const sec of rawSecs) {
    const t = sec.trim();
    if (!t) continue;
    if (!pending) {
      pending = t;
    } else if (pending.length < 150 || t.length < 120) {
      pending = pending + "\n\n" + t;
    } else {
      merged.push(pending);
      pending = t;
    }
  }
  if (pending) merged.push(pending);
  return merged
    .filter((s) => s.length >= 25)
    .map((sec) => {
      const flEnd = sec.indexOf("\n");
      const firstLine = flEnd === -1 ? sec : sec.slice(0, flEnd);
      const heading = firstLine.startsWith("#") ? firstLine.slice(0, 120) : fileTitle;
      const snippet = sec.replace(/\s+/g, " ").slice(0, 220);
      return { heading, snippet, body: sec.slice(0, 1400) };
    });
}

async function syncMarkdownDir(targetDir, indexDir) {
  const absDir = path.resolve(targetDir);
  const files = walkMarkdownFiles(absDir).sort((a, b) => a.sourceId.localeCompare(b.sourceId));
  const byId = new Map(files.map((f) => [f.sourceId, f]));

  return syncHybridIndex(indexDir, {
    extraMeta: { dir: absDir },
    listSources: () => files.map((f) => ({ sourceId: f.sourceId, stamp: f.stamp })),
    loadChunks: (changed) => {
      const chunks = [];
      for (const item of changed) {
        const f = byId.get(item.sourceId);
        if (!f) continue;
        const content = fs.readFileSync(f.fullPath, "utf8");
        const fileTitle = path.basename(f.sourceId, ".md");
        const pathWords = f.sourceId.replace(/[/_.-]/g, " ");
        for (const s of splitMeaningfulSections(content, fileTitle)) {
          chunks.push({
            sourceId: f.sourceId,
            groupKey: f.sourceId,
            h: s.heading,
            s: s.snippet,
            text: `${pathWords}\n${s.heading}\n${s.body}`,
            bm25Text: `${pathWords} ${pathWords}\n${s.heading}\n${s.body}`
          });
        }
      }
      return chunks;
    }
  });
}

async function main() {
  const { values, positionals } = parseArgs({
    options: {
      dir: { type: "string", short: "d" },
      index: { type: "string" },
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

  const targetDir = path.resolve(values.dir || ".");
  const dirKey = fnv1a(targetDir).toString(16).padStart(8, "0");
  const indexDir = values.index
    ? path.resolve(values.index)
    : path.join(os.homedir(), ".cache", "potion-search", `dir-${dirKey}`);

  if (positionals[0] === "sync") {
    const res = await syncMarkdownDir(targetDir, indexDir);
    if (values.pack) {
      packIndexBundle(indexDir, path.resolve(values.pack));
    }
    console.log(JSON.stringify(res));
    return;
  }

  const query = positionals[0] === "search" ? positionals.slice(1).join(" ") : positionals.join(" ");
  if (!query.trim()) {
    process.stderr.write(
      "Usage:\n" +
        '  potion-search "<query>" [--dir <folder>] [--index <index_dir>] [--top 8] [--path <substr>] [--json]\n' +
        "  potion-search sync --dir <folder> [--index <index_dir>] [--pack <bundle.tar.gz>]\n" +
        "  potion-search build-vocab [--out <vocab_dir>] [--pack <archive.tar.gz>]\n"
    );
    process.exit(1);
  }

  if (values.dir || !values.index) {
    const res = await syncMarkdownDir(targetDir, indexDir);
    if (res.updatedSources > 0) {
      process.stderr.write(
        `[potion-search] Synced ${res.chunkCount} chunks (${res.updatedSources}/${res.totalSources} files updated) in ${res.elapsedMs}ms\n`
      );
    }
  }

  const store = await openHybridIndex(indexDir);
  try {
    const pathFilter = values.path ? values.path.toLowerCase() : null;
    const results = store.search(query, {
      topK: Math.max(1, Number(values.top) || 8),
      filterFn: pathFilter ? (c) => c.p.toLowerCase().includes(pathFilter) : null,
      adjustScore: (gk, _c, score) =>
        gk.toLowerCase().includes("deprecated") || gk.toLowerCase().includes("archived/")
          ? score * 0.6
          : score
    });

    if (values.json) {
      console.log(
        JSON.stringify({ indexDir, chunkCount: store.numChunks, results }, null, 2)
      );
      return;
    }

    console.log(`${targetDir} (${store.numChunks} chunks)`);
    for (let i = 0; i < results.length; i++) {
      const r = results[i];
      console.log(
        `\n[${i + 1}] ${r.path}  (rrf=${r.rrfScore}, vec=${r.vecSimilarity ?? "-"}, bm25=${r.bm25Score ?? "-"})`
      );
      console.log(`    ${r.heading}`);
      console.log(`    ${r.snippet}`);
    }
  } finally {
    store.close();
  }
}

const isDirectRun =
  process.argv[1] &&
  fs.existsSync(process.argv[1]) &&
  fileURLToPath(import.meta.url) === fs.realpathSync(process.argv[1]);

if (isDirectRun) {
  main().catch((err) => {
    process.stderr.write(`[potion-search] Error: ${err.message}\n`);
    process.exit(1);
  });
}
