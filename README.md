# potion-search

Zero-dependency Chinese/English hybrid search engine (**256-d Static Int8 Embeddings + `Intl.Segmenter` BM25 + Reciprocal Rank Fusion**) for Node.js >= 20.

## Highlights

- **Zero npm dependencies**: Pure Node.js standard library (`node:fs`, `node:zlib`, `Intl.Segmenter`). No Python, ONNX Runtime, ChromaDB, or native C++ addons.
- **Multilingual Static Embeddings (`minishlab/potion-multilingual-128M`, Plan D)**:
  - Retains all **223,287** Chinese (CJK) and English/ASCII tokens (including `Tokenlearn` phrase expansions) quantized to symmetric Per-Row Int8 (`vectors.i8`).
  - **Pre-compiled Binary FNV-1a Hash Table (`hash.bin`)**: `~2 ms` vocabulary cold-load without building a 223k-entry JS `Map`.
  - **On-demand `pread` (`fs.readSync`)**: Single CLI queries read only the matched token rows (~2.5 KB) directly from disk instead of loading the 55 MB embedding matrix into RAM.
- **Incremental Indexing**:
  - Persists per-chunk tokenized term frequencies (`doc-terms.txt`) and Int8 vectors (`chunks.i8`) so incremental updates only embed and segment newly added or modified documents.
  - Offset-indexed `chunks.jsonl` + `chunks.offsets.u32` ensures queries only `JSON.parse` the final Top-K hits.
- **Auth-free Vocabulary Bootstrap**:
  - Automatically downloads `potion-vocab-v2.tar.gz` (~53 MB) from this public repository's GitHub Releases via built-in `fetch()` on first run (cached in `~/.cache/xc-potion-vocab/v2`), with automatic fallback to compiling from HuggingFace `minishlab/potion-multilingual-128M`.

## CLI Usage

```bash
# Search any Markdown directory (incrementally indexes on first run or file change)
node ./bin/potion-search.mjs "WebView2 Mac crash dump 调试" --dir /path/to/notes --top 8

# Filter by path substring or output JSON
node ./bin/potion-search.mjs "DNS DoH 证书" --dir /path/to/notes --path "network/" --json

# Rebuild vocabulary archive from raw HuggingFace weights
node ./bin/potion-search.mjs build-vocab --pack ./potion-vocab-v2.tar.gz
```

## Library API

```js
import {
  ensureVocab,
  loadVocab,
  embedText,
  quantizeRow,
  encodeChunkTerms,
  buildBm25FromEncodedTerms,
  writeIndexFiles,
  loadIndexFiles,
  queryHybridIndex
} from "potion-search";
```

## License

MIT (distilled vocabulary derived from [`minishlab/potion-multilingual-128M`](https://huggingface.co/minishlab/potion-multilingual-128M), MIT License).
