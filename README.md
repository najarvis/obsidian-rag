# obsidian-rag

MCP server that generates local embeddings for documents or text using a deterministic, dependency-free feature-hashing (hashing trick) technique and stores them in a JSONL file.

## Setup

```bash
uv sync
```

Embeddings are computed in pure Python (via `hashlib`), so there is no model to download.

## Cursor MCP configuration

Add to `~/.cursor/mcp.json` (adjust the path as needed):

```json
{
  "mcpServers": {
    "obsidian-rag": {
      "command": "uv",
      "args": ["run", "--directory", "/home/jarvis/workspace/obsidian-rag", "obsidian-rag-mcp"]
    }
  }
}
```

Restart Cursor after saving the config.

## Tool: `embed_and_store`

| Parameter   | Required | Description |
|-------------|----------|-------------|
| `reference` | yes      | Absolute file path or arbitrary reference string |
| `text`      | no       | Inline text to embed; if omitted, `reference` is read as a file path |

Embeddings are appended to `data/embeddings.jsonl` by default.

## Tool: `index_directory`

Embed and store every markdown (`.md`) file under a directory (recursive). Files whose content is already present in the store are skipped, so re-runs are idempotent.

| Parameter   | Required | Description |
|-------------|----------|-------------|
| `directory` | yes      | Absolute path to the directory to index |

Returns a JSON summary with `indexed`, `skipped`, and `total` counts.

## Environment variables

| Variable               | Default                  | Description |
|------------------------|--------------------------|-------------|
| `EMBEDDINGS_STORE_PATH`| `data/embeddings.jsonl`  | Path to the JSONL store |
| `EMBEDDING_DIM`        | `256`                    | Dimensionality of the feature-hash vectors |

## Indexing a directory

Embed every `.md` file under a directory (recursive) and append the records to the store. Files already present in the store (by content hash) are skipped:

```bash
uv run python -m obsidian_rag.indexer /path/to/vault
```
