/**
 * potion-search — Zero-dependency Chinese/English hybrid search engine for Node.js >= 20.
 *
 * Combines:
 *   - 256-d symmetric Int8 static embeddings from `minishlab/potion-multilingual-128M` (Plan D: 223k CJK+ASCII tokens)
 *   - Pre-compiled binary FNV-1a open-addressing hash table (`hash.bin`) + on-demand `pread` (`vectors.i8`)
 *   - CJK (`Intl.Segmenter('zh')` + bigrams) & English stemmed Okapi BM25 with incremental per-chunk term cache
 *   - Reciprocal Rank Fusion (RRF) + offset-indexed lazy JSONL chunk decoding
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";

export const DIM = 256;
export const DEFAULT_VOCAB_RELEASE_URL =
  process.env.POTION_VOCAB_URL ||
  "https://github.com/billxc/potion-search/releases/download/vocab-v2/potion-vocab-v2.tar.gz";

const HF_MODEL_ID = "minishlab/potion-multilingual-128M";
const HF_BASE_URL = `https://huggingface.co/${HF_MODEL_ID}/resolve/main`;
const HASH_CAP = 1 << 19; // 524,288 slots for ~223k tokens (<45% load factor)
const HASH_MASK = HASH_CAP - 1;

const STOP_WORDS = new Set(
  "the a an and or of to in for with on by is are be as it this that from at use using your you can how when which into not".split(
    " "
  )
);

const zhSegmenter =
  typeof Intl !== "undefined" && typeof Intl.Segmenter === "function"
    ? new Intl.Segmenter("zh", { granularity: "word" })
    : null;

export function getVocabCacheDir() {
  return (
    process.env.XC_POTION_VOCAB_DIR ||
    process.env.POTION_VOCAB_DIR ||
    path.join(os.homedir(), ".cache", "xc-potion-vocab", "v2")
  );
}

export function fnv1a(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function writeOctal(buf, offset, length, value) {
  const oct = Math.max(0, value).toString(8).padStart(length - 1, "0");
  buf.write(oct.slice(0, length - 1), offset, length - 1, "ascii");
  buf[offset + length - 1] = 0;
}

export function packTarGzFlat(sourceDir, fileNames, outArchivePath) {
  const blocks = [];
  for (const name of fileNames) {
    const safeName = path.posix.basename(name.replace(/\\/g, "/"));
    const fullPath = path.join(sourceDir, safeName);
    const data = fs.readFileSync(fullPath);
    const stat = fs.statSync(fullPath);

    const header = Buffer.alloc(512, 0);
    header.write(safeName, 0, Math.min(100, Buffer.byteLength(safeName)), "utf8");
    writeOctal(header, 100, 8, 0o644);
    writeOctal(header, 108, 8, 0);
    writeOctal(header, 116, 8, 0);
    writeOctal(header, 124, 12, data.length);
    writeOctal(header, 136, 12, Math.floor(stat.mtimeMs / 1000));
    header.fill(0x20, 148, 156);
    header[156] = "0".charCodeAt(0);
    header.write("ustar\0", 257, 6, "ascii");
    header.write("00", 263, 2, "ascii");

    let sum = 0;
    for (let i = 0; i < 512; i++) sum += header[i];
    header.write(sum.toString(8).padStart(6, "0"), 148, 6, "ascii");
    header[154] = 0;
    header[155] = 0x20;

    blocks.push(header, data);
    const rem = data.length % 512;
    if (rem !== 0) blocks.push(Buffer.alloc(512 - rem, 0));
  }
  blocks.push(Buffer.alloc(1024, 0));
  const gzBuf = zlib.gzipSync(Buffer.concat(blocks), { level: 9 });
  fs.mkdirSync(path.dirname(outArchivePath), { recursive: true });
  fs.writeFileSync(outArchivePath, gzBuf);
}

export function extractTarGzFlat(archivePath, destDir) {
  const tarBuf = zlib.gunzipSync(fs.readFileSync(archivePath));
  fs.mkdirSync(destDir, { recursive: true });
  let offset = 0;

  while (offset + 512 <= tarBuf.length) {
    const header = tarBuf.subarray(offset, offset + 512);
    offset += 512;
    let allZero = true;
    for (let i = 0; i < 512; i++) {
      if (header[i] !== 0) {
        allZero = false;
        break;
      }
    }
    if (allZero) break;

    const rawName = header.subarray(0, 100).toString("utf8").replace(/\0.*$/, "");
    const sizeOctal = header.subarray(124, 136).toString("ascii").replace(/\0.*$/, "").trim();
    const size = sizeOctal ? parseInt(sizeOctal, 8) : 0;
    const typeFlag = header[156] === 0 ? "0" : String.fromCharCode(header[156]);
    const paddedSize = Math.ceil(size / 512) * 512;
    const body = tarBuf.subarray(offset, offset + size);
    offset += paddedSize;

    const baseName = path.posix.basename(rawName.replace(/\\/g, "/"));
    if (typeFlag === "0" && baseName && !baseName.startsWith(".")) {
      fs.writeFileSync(path.join(destDir, baseName), body);
    }
  }
}

export function hasCompleteVocab(vocabDir = getVocabCacheDir()) {
  return (
    fs.existsSync(path.join(vocabDir, "tokens.bin")) &&
    fs.existsSync(path.join(vocabDir, "hash.bin")) &&
    fs.existsSync(path.join(vocabDir, "scores.f32")) &&
    fs.existsSync(path.join(vocabDir, "vectors.i8")) &&
    fs.existsSync(path.join(vocabDir, "scales.f32"))
  );
}

async function downloadFile(url, destPath) {
  fs.mkdirSync(path.dirname(destPath), { recursive: true });
  const tmpPath = `${destPath}.${process.pid}.tmp`;
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} fetching ${url}`);
  }
  const ab = await res.arrayBuffer();
  fs.writeFileSync(tmpPath, Buffer.from(ab));
  fs.renameSync(tmpPath, destPath);
}

async function tryDownloadVocabFromPublicRelease(vocabDir, releaseUrl = DEFAULT_VOCAB_RELEASE_URL) {
  const stageDir = `${vocabDir}.rel-${process.pid}`;
  const archivePath = path.join(stageDir, "potion-vocab-v2.tar.gz");
  const unpackDir = path.join(stageDir, "unpacked");

  try {
    await downloadFile(releaseUrl, archivePath);
    extractTarGzFlat(archivePath, unpackDir);
    if (!hasCompleteVocab(unpackDir)) return false;

    fs.mkdirSync(path.dirname(vocabDir), { recursive: true });
    if (fs.existsSync(vocabDir)) {
      fs.rmSync(vocabDir, { recursive: true, force: true });
    }
    fs.renameSync(unpackDir, vocabDir);
    return true;
  } catch {
    return false;
  } finally {
    fs.rmSync(stageDir, { recursive: true, force: true });
  }
}

function findLocalHfSnapshot() {
  const hfRoot = path.join(
    os.homedir(),
    ".cache",
    "huggingface",
    "hub",
    "models--minishlab--potion-multilingual-128M",
    "snapshots"
  );
  if (!fs.existsSync(hfRoot)) return null;
  for (const entry of fs.readdirSync(hfRoot)) {
    const dir = path.join(hfRoot, entry);
    const tokPath = path.join(dir, "tokenizer.json");
    const stPath = path.join(dir, "model.safetensors");
    if (fs.existsSync(tokPath) && fs.existsSync(stPath)) {
      return { tokPath, stPath };
    }
  }
  return null;
}

export async function buildVocabFromHf(vocabDir = getVocabCacheDir()) {
  let paths = findLocalHfSnapshot();
  let tempRawDir = null;
  try {
    if (!paths) {
      tempRawDir = path.join(path.dirname(vocabDir), `.hf-tmp-${process.pid}`);
      const tokPath = path.join(tempRawDir, "tokenizer.json");
      const stPath = path.join(tempRawDir, "model.safetensors");
      process.stderr.write(`[potion-search] Downloading ${HF_MODEL_ID}/tokenizer.json ...\n`);
      await downloadFile(`${HF_BASE_URL}/tokenizer.json`, tokPath);
      process.stderr.write(`[potion-search] Downloading ${HF_MODEL_ID}/model.safetensors (~512MB) ...\n`);
      await downloadFile(`${HF_BASE_URL}/model.safetensors`, stPath);
      paths = { tokPath, stPath };
    }

    process.stderr.write(`[potion-search] Compiling Plan-D (223k zh+en) binary vocabulary in ${vocabDir} ...\n`);
    const tok = JSON.parse(fs.readFileSync(paths.tokPath, "utf8"));
    const stBuf = fs.readFileSync(paths.stPath);
    const hdrLen = Number(stBuf.readBigUInt64LE(0));
    const hdr = JSON.parse(stBuf.subarray(8, 8 + hdrLen).toString("utf8"));
    const tensorMeta = hdr.embeddings || Object.values(hdr).find((v) => v?.shape?.[1] === DIM);
    const [totalRows] = tensorMeta.shape;
    const [byteStart] = tensorMeta.data_offsets;
    const rawF32 = new Float32Array(
      stBuf.buffer,
      stBuf.byteOffset + 8 + hdrLen + byteStart,
      totalRows * DIM
    );

    const keepRe = /^[▁a-zA-Z0-9_\-./:@+#一-鿿]+$/;
    const rawVocab = tok.model.vocab;
    const keptPieces = [];
    const keptScores = [];
    const keptIdx = [];

    for (let i = 0; i < rawVocab.length; i++) {
      const [piece, score] = rawVocab[i];
      if (piece.startsWith("[") && piece.endsWith("]")) continue;
      if (keepRe.test(piece)) {
        keptPieces.push(piece);
        keptScores.push(score);
        keptIdx.push(i);
      }
    }

    const numTokens = keptIdx.length;
    const scoresF32 = new Float32Array(keptScores);
    const vectorsI8 = new Int8Array(numTokens * DIM);
    const scalesF32 = new Float32Array(numTokens);

    const hashKeys = new Int32Array(HASH_CAP).fill(-1);
    const hashIds = new Int32Array(HASH_CAP).fill(-1);
    const offsetsU32 = new Uint32Array(numTokens + 1);
    const pieceBuffers = new Array(numTokens);
    let byteCursor = 0;

    for (let id = 0; id < numTokens; id++) {
      const piece = keptPieces[id];
      const pBuf = Buffer.from(piece, "utf8");
      pieceBuffers[id] = pBuf;
      offsetsU32[id] = byteCursor;
      byteCursor += pBuf.byteLength;

      const h = fnv1a(piece);
      let slot = h & HASH_MASK;
      while (hashIds[slot] !== -1) slot = (slot + 1) & HASH_MASK;
      hashKeys[slot] = h | 0;
      hashIds[slot] = id;

      const srcBase = keptIdx[id] * DIM;
      let maxAbs = 0;
      for (let d = 0; d < DIM; d++) {
        const v = Math.abs(rawF32[srcBase + d]);
        if (v > maxAbs) maxAbs = v;
      }
      const scale = maxAbs / 127 || 1;
      scalesF32[id] = scale;
      const dstBase = id * DIM;
      for (let d = 0; d < DIM; d++) {
        vectorsI8[dstBase + d] = Math.max(
          -127,
          Math.min(127, Math.round(rawF32[srcBase + d] / scale))
        );
      }
    }
    offsetsU32[numTokens] = byteCursor;

    const hashBin = Buffer.allocUnsafe(
      hashKeys.byteLength + hashIds.byteLength + offsetsU32.byteLength
    );
    Buffer.from(hashKeys.buffer).copy(hashBin, 0);
    Buffer.from(hashIds.buffer).copy(hashBin, hashKeys.byteLength);
    Buffer.from(offsetsU32.buffer).copy(hashBin, hashKeys.byteLength + hashIds.byteLength);

    const stageDir = `${vocabDir}.tmp-${process.pid}`;
    fs.mkdirSync(stageDir, { recursive: true });
    fs.writeFileSync(path.join(stageDir, "tokens.bin"), Buffer.concat(pieceBuffers));
    fs.writeFileSync(path.join(stageDir, "hash.bin"), hashBin);
    fs.writeFileSync(path.join(stageDir, "scores.f32"), Buffer.from(scoresF32.buffer));
    fs.writeFileSync(path.join(stageDir, "vectors.i8"), Buffer.from(vectorsI8.buffer));
    fs.writeFileSync(path.join(stageDir, "scales.f32"), Buffer.from(scalesF32.buffer));

    fs.mkdirSync(path.dirname(vocabDir), { recursive: true });
    if (fs.existsSync(vocabDir)) {
      fs.rmSync(vocabDir, { recursive: true, force: true });
    }
    fs.renameSync(stageDir, vocabDir);
    return vocabDir;
  } finally {
    if (tempRawDir) {
      fs.rmSync(tempRawDir, { recursive: true, force: true });
    }
  }
}

/**
 * Ensure binary vocabulary exists locally:
 *   1. Local cache `~/.cache/xc-potion-vocab/v2` -> 0ms
 *   2. Auth-free public GitHub Release download (`billxc/potion-search`) -> ~53MB `.tar.gz` via `fetch()`
 *   3. Fallback to HuggingFace raw weights compilation
 */
export async function ensureVocab(vocabDir = getVocabCacheDir()) {
  if (hasCompleteVocab(vocabDir)) return vocabDir;

  process.stderr.write(`[potion-search] Fetching vocabulary from ${DEFAULT_VOCAB_RELEASE_URL} ...\n`);
  if (await tryDownloadVocabFromPublicRelease(vocabDir)) {
    return vocabDir;
  }

  return buildVocabFromHf(vocabDir);
}

let cachedVocab = null;

export function loadVocab(vocabDir = getVocabCacheDir(), { preloadVectors = false } = {}) {
  if (cachedVocab && cachedVocab.vocabDir === vocabDir) {
    if (preloadVectors && !cachedVocab.table) {
      const i8Buf = fs.readFileSync(path.join(vocabDir, "vectors.i8"));
      cachedVocab.table = new Int8Array(i8Buf.buffer, i8Buf.byteOffset, i8Buf.byteLength);
    }
    return cachedVocab;
  }

  const tokensBuf = fs.readFileSync(path.join(vocabDir, "tokens.bin"));
  const hashBuf = fs.readFileSync(path.join(vocabDir, "hash.bin"));
  const alignedHash = hashBuf.buffer.slice(
    hashBuf.byteOffset,
    hashBuf.byteOffset + hashBuf.byteLength
  );

  const hashKeys = new Int32Array(alignedHash, 0, HASH_CAP);
  const hashIds = new Int32Array(alignedHash, HASH_CAP * 4, HASH_CAP);
  const offsets = new Uint32Array(alignedHash, HASH_CAP * 8);

  const scoresBuf = fs.readFileSync(path.join(vocabDir, "scores.f32"));
  const scores = new Float32Array(
    scoresBuf.buffer.slice(scoresBuf.byteOffset, scoresBuf.byteOffset + scoresBuf.byteLength)
  );

  const scalesBuf = fs.readFileSync(path.join(vocabDir, "scales.f32"));
  const scales = new Float32Array(
    scalesBuf.buffer.slice(scalesBuf.byteOffset, scalesBuf.byteOffset + scalesBuf.byteLength)
  );

  let table = null;
  const vecPath = path.join(vocabDir, "vectors.i8");
  const vecFd = fs.openSync(vecPath, "r");
  if (preloadVectors) {
    const i8Buf = fs.readFileSync(vecPath);
    table = new Int8Array(i8Buf.buffer, i8Buf.byteOffset, i8Buf.byteLength);
  }

  function lookupTokenId(str) {
    const h = fnv1a(str);
    const hSigned = h | 0;
    let slot = h & HASH_MASK;
    while (true) {
      const id = hashIds[slot];
      if (id === -1) return undefined;
      if (hashKeys[slot] === hSigned) {
        const sOff = offsets[id];
        const eOff = offsets[id + 1];
        if (tokensBuf.toString("utf8", sOff, eOff) === str) return id;
      }
      slot = (slot + 1) & HASH_MASK;
    }
  }

  cachedVocab = {
    vocabDir,
    lookupTokenId,
    scores,
    scales,
    table,
    vecFd,
    rowScratch: Buffer.allocUnsafe(DIM),
    spanCache: new Map()
  };
  return cachedVocab;
}

function tokenizeSpan(raw, isCjk, vocab) {
  const { lookupTokenId, scores, spanCache } = vocab;
  const cached = spanCache.get(raw);
  if (cached !== undefined) return cached;

  const withPrefix = "▁" + raw;
  const exact = lookupTokenId(withPrefix) ?? lookupTokenId(raw);
  if (exact !== undefined) {
    const res = [exact];
    if (raw.length <= 24) spanCache.set(raw, res);
    return res;
  }

  const lower = raw.toLowerCase();
  if (lower !== raw) {
    const exactLower = lookupTokenId("▁" + lower) ?? lookupTokenId(lower);
    if (exactLower !== undefined) {
      const res = [exactLower];
      if (raw.length <= 24) spanCache.set(raw, res);
      return res;
    }
  }

  const str = isCjk ? raw : withPrefix;
  const maxStep = isCjk ? 6 : 16;
  const n = str.length;
  const bestScore = new Float64Array(n + 1).fill(-Infinity);
  const bestPrev = new Int32Array(n + 1).fill(-1);
  const bestId = new Int32Array(n + 1).fill(-1);
  bestScore[0] = 0;

  for (let i = 0; i < n; i++) {
    if (bestScore[i] === -Infinity) continue;
    const limit = Math.min(n, i + maxStep);
    let matched = false;
    for (let j = i + 1; j <= limit; j++) {
      const sub = str.slice(i, j);
      let id =
        i === 0 && isCjk
          ? lookupTokenId("▁" + sub) ?? lookupTokenId(sub)
          : lookupTokenId(sub);
      if (id === undefined && !isCjk) {
        id = lookupTokenId(sub.toLowerCase());
      }
      if (id !== undefined) {
        matched = true;
        const cand = bestScore[i] + scores[id];
        if (cand > bestScore[j]) {
          bestScore[j] = cand;
          bestPrev[j] = i;
          bestId[j] = id;
        }
      }
    }
    if (!matched && bestScore[i] - 20 > bestScore[i + 1]) {
      bestScore[i + 1] = bestScore[i] - 20;
      bestPrev[i + 1] = i;
      bestId[i + 1] = -1;
    }
  }

  const ids = [];
  let cur = n;
  while (cur > 0 && bestPrev[cur] !== -1) {
    if (bestId[cur] !== -1) ids.push(bestId[cur]);
    cur = bestPrev[cur];
  }
  ids.reverse();
  if (raw.length <= 24 && spanCache.size < 200000) {
    spanCache.set(raw, ids);
  }
  return ids;
}

const SPAN_RE = /[a-zA-Z0-9_]+|[一-鿿]+/g;

export function embedText(text, vocab, out = new Float32Array(DIM)) {
  out.fill(0);
  const { table, scales, vecFd, rowScratch } = vocab;
  const matches = text.match(SPAN_RE);
  if (!matches) return out;

  const rowI8 = table
    ? null
    : new Int8Array(rowScratch.buffer, rowScratch.byteOffset, DIM);

  for (let p = 0; p < matches.length; p++) {
    const raw = matches[p];
    const isCjk = raw.charCodeAt(0) >= 0x4e00;
    const ids = tokenizeSpan(raw, isCjk, vocab);
    for (let k = 0; k < ids.length; k++) {
      const id = ids[k];
      const s = scales[id];
      if (table) {
        const base = id * DIM;
        for (let d = 0; d < DIM; d++) out[d] += s * table[base + d];
      } else {
        fs.readSync(vecFd, rowScratch, 0, DIM, id * DIM);
        for (let d = 0; d < DIM; d++) out[d] += s * rowI8[d];
      }
    }
  }

  let normSq = 0;
  for (let d = 0; d < DIM; d++) normSq += out[d] * out[d];
  const norm = Math.sqrt(normSq) || 1;
  for (let d = 0; d < DIM; d++) out[d] /= norm;
  return out;
}

export function quantizeRow(vecF32, outI8, rowIdx, outScales) {
  let maxAbs = 0;
  for (let d = 0; d < DIM; d++) {
    const a = Math.abs(vecF32[d]);
    if (a > maxAbs) maxAbs = a;
  }
  const scale = maxAbs / 127 || 1;
  outScales[rowIdx] = scale;
  const base = rowIdx * DIM;
  for (let d = 0; d < DIM; d++) {
    outI8[base + d] = Math.max(-127, Math.min(127, Math.round(vecF32[d] / scale)));
  }
}

export function stemEnglishWord(word) {
  const len = word.length;
  if (len < 3 || word[len - 1] !== "s") return word;
  if (word[len - 2] === "u" || word[len - 2] === "s") return word;
  if (word[len - 2] === "e") {
    if (len > 3 && word[len - 3] === "i" && word[len - 4] !== "a" && word[len - 4] !== "e") {
      return word.slice(0, len - 3) + "y";
    }
    if ("iaoe".includes(word[len - 3])) return word;
    return word.slice(0, len - 1);
  }
  return word.slice(0, len - 1);
}

export function tokenizeBm25(text) {
  const lower = text.toLowerCase();
  const out = [];

  const enMatches = lower.match(/[a-z0-9][a-z0-9_-]+/g);
  if (enMatches) {
    for (let i = 0; i < enMatches.length; i++) {
      const w = enMatches[i];
      if (!STOP_WORDS.has(w)) out.push(stemEnglishWord(w));
    }
  }

  const zhMatches = lower.match(/[一-鿿]+/g);
  if (zhMatches) {
    for (let i = 0; i < zhMatches.length; i++) {
      const run = zhMatches[i];
      if (zhSegmenter) {
        for (const seg of zhSegmenter.segment(run)) {
          if (seg.segment) out.push(seg.segment);
        }
      }
      for (let k = 0; k < run.length; k++) {
        if (!zhSegmenter) out.push(run[k]);
        if (k + 1 < run.length) out.push(run.slice(k, k + 2));
      }
    }
  }

  return out;
}

export function encodeChunkTerms(text) {
  const toks = tokenizeBm25(text);
  const len = Math.min(65535, toks.length);
  const tfMap = new Map();
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i];
    tfMap.set(t, (tfMap.get(t) || 0) + 1);
  }
  const parts = [String(len)];
  for (const [term, tf] of tfMap) {
    parts.push(`${term}:${Math.min(255, tf)}`);
  }
  return parts.join("\t");
}

export function buildBm25FromEncodedTerms(encodedTermLines) {
  const numChunks = encodedTermLines.length;
  const docLens = new Uint16Array(numChunks);
  const postingsMap = new Map();
  let totalLen = 0;

  for (let i = 0; i < numChunks; i++) {
    const line = encodedTermLines[i];
    const firstTab = line.indexOf("\t");
    const len = Number(firstTab === -1 ? line : line.slice(0, firstTab)) || 0;
    docLens[i] = len;
    totalLen += len;
    if (firstTab === -1) continue;

    let pos = firstTab + 1;
    while (pos < line.length) {
      let nextTab = line.indexOf("\t", pos);
      if (nextTab === -1) nextTab = line.length;
      const colon = line.lastIndexOf(":", nextTab - 1);
      const term = line.slice(pos, colon);
      const tf = Number(line.slice(colon + 1, nextTab)) || 1;
      let list = postingsMap.get(term);
      if (!list) {
        list = [];
        postingsMap.set(term, list);
      }
      list.push((i << 8) | tf);
      pos = nextTab + 1;
    }
  }

  const avgLen = totalLen / numChunks || 1;
  let totalPostings = 0;
  for (const list of postingsMap.values()) totalPostings += list.length;

  const postingsU32 = new Uint32Array(totalPostings);
  const lines = [`#avgLen=${avgLen}`];
  let cursor = 0;
  for (const [term, list] of postingsMap) {
    lines.push(`${term}\t${cursor}\t${list.length}`);
    for (let j = 0; j < list.length; j++) postingsU32[cursor++] = list[j];
  }

  const rawTerms = "\n" + lines.join("\n") + "\n";
  const rawBin = Buffer.allocUnsafe(docLens.byteLength + postingsU32.byteLength);
  Buffer.from(docLens.buffer, docLens.byteOffset, docLens.byteLength).copy(rawBin, 0);
  Buffer.from(postingsU32.buffer, postingsU32.byteOffset, postingsU32.byteLength).copy(
    rawBin,
    docLens.byteLength
  );

  return { avgLen, rawTerms, rawBin };
}

export function buildBm25InvertedBuffers(textsForBm25) {
  return buildBm25FromEncodedTerms(textsForBm25.map(encodeChunkTerms));
}

export function writeIndexFiles(indexDir, metaObj, vecI8, vecScales, bm25, encodedTermLines = null) {
  const stageDir = `${indexDir}.tmp-${process.pid}`;
  fs.mkdirSync(stageDir, { recursive: true });

  const { chunks, ...headerMeta } = metaObj;
  headerMeta.chunkCount = chunks.length;

  const offsetsU32 = new Uint32Array(chunks.length + 1);
  const lineBuffers = new Array(chunks.length);
  let byteCursor = 0;
  for (let i = 0; i < chunks.length; i++) {
    const buf = Buffer.from(JSON.stringify(chunks[i]) + "\n", "utf8");
    lineBuffers[i] = buf;
    offsetsU32[i] = byteCursor;
    byteCursor += buf.byteLength;
  }
  offsetsU32[chunks.length] = byteCursor;

  fs.writeFileSync(path.join(stageDir, "meta.json"), JSON.stringify(headerMeta), "utf8");
  fs.writeFileSync(path.join(stageDir, "chunks.jsonl"), Buffer.concat(lineBuffers));
  fs.writeFileSync(
    path.join(stageDir, "chunks.offsets.u32"),
    Buffer.from(offsetsU32.buffer, offsetsU32.byteOffset, offsetsU32.byteLength)
  );
  fs.writeFileSync(
    path.join(stageDir, "chunks.i8"),
    Buffer.from(vecI8.buffer, vecI8.byteOffset, vecI8.byteLength)
  );
  fs.writeFileSync(
    path.join(stageDir, "chunks.scale.f32"),
    Buffer.from(vecScales.buffer, vecScales.byteOffset, vecScales.byteLength)
  );
  fs.writeFileSync(path.join(stageDir, "bm25.txt"), bm25.rawTerms, "utf8");
  fs.writeFileSync(path.join(stageDir, "bm25.bin"), bm25.rawBin);
  if (encodedTermLines) {
    fs.writeFileSync(path.join(stageDir, "doc-terms.txt"), encodedTermLines.join("\n"), "utf8");
  }

  fs.mkdirSync(path.dirname(indexDir), { recursive: true });
  if (fs.existsSync(indexDir)) {
    fs.rmSync(indexDir, { recursive: true, force: true });
  }
  fs.renameSync(stageDir, indexDir);
}

export function hasIndexFiles(indexDir) {
  return (
    fs.existsSync(path.join(indexDir, "meta.json")) &&
    fs.existsSync(path.join(indexDir, "chunks.jsonl")) &&
    fs.existsSync(path.join(indexDir, "chunks.offsets.u32")) &&
    fs.existsSync(path.join(indexDir, "chunks.i8")) &&
    fs.existsSync(path.join(indexDir, "chunks.scale.f32")) &&
    fs.existsSync(path.join(indexDir, "bm25.txt")) &&
    fs.existsSync(path.join(indexDir, "bm25.bin"))
  );
}

export function loadIndexFiles(indexDir, { lazyChunks = true, loadDocTerms = false } = {}) {
  const meta = JSON.parse(fs.readFileSync(path.join(indexDir, "meta.json"), "utf8"));
  const numChunks = meta.chunkCount;

  const offBuf = fs.readFileSync(path.join(indexDir, "chunks.offsets.u32"));
  const offsets = new Uint32Array(
    offBuf.buffer.slice(offBuf.byteOffset, offBuf.byteOffset + offBuf.byteLength)
  );
  const chunksJsonlPath = path.join(indexDir, "chunks.jsonl");

  let chunks = null;
  let chunksFd = null;
  const parsedCache = new Map();

  if (!lazyChunks) {
    const raw = fs.readFileSync(chunksJsonlPath, "utf8");
    chunks = new Array(numChunks);
    let pos = 0;
    for (let i = 0; i < numChunks; i++) {
      const nl = raw.indexOf("\n", pos);
      chunks[i] = JSON.parse(raw.slice(pos, nl));
      pos = nl + 1;
    }
  } else {
    chunksFd = fs.openSync(chunksJsonlPath, "r");
  }

  function getChunk(idx) {
    if (chunks) return chunks[idx];
    let c = parsedCache.get(idx);
    if (c !== undefined) return c;
    const sOff = offsets[idx];
    const len = offsets[idx + 1] - sOff;
    const buf = Buffer.allocUnsafe(len);
    fs.readSync(chunksFd, buf, 0, len, sOff);
    c = JSON.parse(buf.toString("utf8"));
    parsedCache.set(idx, c);
    return c;
  }

  const i8Buf = fs.readFileSync(path.join(indexDir, "chunks.i8"));
  const vecI8 = new Int8Array(i8Buf.buffer, i8Buf.byteOffset, i8Buf.byteLength);

  const scaleBuf = fs.readFileSync(path.join(indexDir, "chunks.scale.f32"));
  const vecScales = new Float32Array(
    scaleBuf.buffer.slice(scaleBuf.byteOffset, scaleBuf.byteOffset + scaleBuf.byteLength)
  );

  const rawTerms = fs.readFileSync(path.join(indexDir, "bm25.txt"), "utf8");
  const firstNl = rawTerms.indexOf("\n", 1);
  const avgLen = Number(rawTerms.slice("\n#avgLen=".length, firstNl)) || 1;
  const rawBin = fs.readFileSync(path.join(indexDir, "bm25.bin"));

  let docTerms = null;
  if (loadDocTerms) {
    const dtPath = path.join(indexDir, "doc-terms.txt");
    if (fs.existsSync(dtPath)) {
      docTerms = fs.readFileSync(dtPath, "utf8").split("\n");
    }
  }

  return {
    meta,
    chunks,
    getChunk,
    numChunks,
    vecI8,
    vecScales,
    avgLen,
    rawTerms,
    rawBin,
    docTerms
  };
}

export function queryHybridIndex(
  query,
  index,
  vocab,
  { filterFn = null, maxRank = 150 } = {}
) {
  const { numChunks, vecI8, vecScales, avgLen, rawTerms, rawBin, getChunk } = index;
  const qVec = embedText(query, vocab);

  const vecScores = new Float64Array(numChunks);
  for (let i = 0; i < numChunks; i++) {
    if (filterFn && !filterFn(getChunk(i), i)) {
      vecScores[i] = -Infinity;
      continue;
    }
    const base = i * DIM;
    let dot = 0;
    for (let d = 0; d < DIM; d++) {
      dot += qVec[d] * vecI8[base + d];
    }
    vecScores[i] = dot * vecScales[i];
  }

  const vecRanked = [];
  if (numChunks > 4096 && !filterFn) {
    const sample = [];
    for (let i = 0; i < numChunks; i += 32) sample.push(vecScores[i]);
    sample.sort((a, b) => b - a);
    const thresh = sample[Math.min(sample.length - 1, 64)] ?? -Infinity;
    for (let i = 0; i < numChunks; i++) {
      if (vecScores[i] >= thresh) vecRanked.push({ idx: i, score: vecScores[i] });
    }
  } else {
    for (let i = 0; i < numChunks; i++) {
      if (vecScores[i] > -Infinity) vecRanked.push({ idx: i, score: vecScores[i] });
    }
  }
  vecRanked.sort((a, b) => b.score - a.score || a.idx - b.idx);
  if (vecRanked.length > maxRank) vecRanked.length = maxRank;

  const qTerms = [...new Set(tokenizeBm25(query))];
  const bm25Ranked = [];

  if (qTerms.length > 0 && numChunks > 0) {
    const firstNl = rawTerms.indexOf("\n", 1);
    const postBaseOffset = numChunks * 2;
    const scoreByDoc = new Float64Array(numChunks);
    const hitDocs = [];

    for (const qt of qTerms) {
      const needle = `\n${qt}\t`;
      const pos = rawTerms.indexOf(needle, firstNl);
      if (pos === -1) continue;
      const valStart = pos + needle.length;
      const tabPos = rawTerms.indexOf("\t", valStart);
      const lineEnd = rawTerms.indexOf("\n", tabPos);
      const start = Number(rawTerms.slice(valStart, tabPos));
      const df = Number(rawTerms.slice(tabPos + 1, lineEnd));
      if (!(df > 0)) continue;

      const idf = Math.log(1 + (numChunks - df + 0.5) / (df + 0.5));
      let byteOff = postBaseOffset + start * 4;
      for (let p = 0; p < df; p++, byteOff += 4) {
        const packed = rawBin.readUInt32LE(byteOff);
        const docIdx = packed >>> 8;
        if (filterFn && vecScores[docIdx] === -Infinity) continue;
        const tf = packed & 0xff;
        const docLen = rawBin.readUInt16LE(docIdx * 2);
        if (scoreByDoc[docIdx] === 0) hitDocs.push(docIdx);
        scoreByDoc[docIdx] +=
          (idf * (tf * 2.2)) / (tf + 1.2 * (0.25 + (0.75 * docLen) / avgLen));
      }
    }

    for (let i = 0; i < hitDocs.length; i++) {
      const idx = hitDocs[i];
      bm25Ranked.push({ idx, score: scoreByDoc[idx] });
    }
    bm25Ranked.sort((a, b) => b.score - a.score || a.idx - b.idx);
    if (bm25Ranked.length > maxRank) bm25Ranked.length = maxRank;
  }

  return { vecRanked, bm25Ranked, vecScores };
}
