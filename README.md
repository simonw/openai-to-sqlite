# openai-to-sqlite

[![PyPI](https://img.shields.io/pypi/v/openai-to-sqlite.svg)](https://pypi.org/project/openai-to-sqlite/)
[![Changelog](https://img.shields.io/github/v/release/simonw/openai-to-sqlite?include_prereleases&label=changelog)](https://github.com/simonw/openai-to-sqlite/releases)
[![Tests](https://github.com/simonw/openai-to-sqlite/workflows/Test/badge.svg)](https://github.com/simonw/openai-to-sqlite/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/openai-to-sqlite/blob/master/LICENSE)

This tool provides utilities for interacting with OpenAI APIs and storing the results in a SQLite database.

## Installation

Install this tool using `pip`:

    pip install openai-to-sqlite

## Configuration

You will need an OpenAI API key to use this tool.

You can create one at https://beta.openai.com/account/api-keys

You can then either set the API key as an environment variable:

    export OPENAI_API_KEY=sk-...

Or pass it to each command using the `--token sk-...` option.

## Embeddings

The `embeddings` command can be used to calculate and store [OpenAI embeddings](https://beta.openai.com/docs/guides/embeddings) for strings of text.

Each embedding has a cost, so be sure to familiarize yourself with [the pricing](https://openai.com/api/pricing/) for the embedding model.

The command can accept data in four different ways:

- As a JSON file containing a list of objects
- As a CSV file
- As a TSV file
- By running queries against a SQLite database

For all of these formats there should be an `id` column, followed by one or more text columns.

The ID will be stored as the content ID. Any other columns will be concatenated together and used as the text to be embedded.

The embeddings from the API will then be saved as binary blobs in the `embeddings` table of the specified SQLite database - or another table, if you pass the `-t/--table` option.

### JSON, CSV and TSV

Given a CSV file like this:

    id,content
    1,This is a test
    2,This is another test

Embeddings can be stored like so:
```bash
openai-to-sqlite embeddings embeddings.db data.csv
```

The resulting schema looks like this:

```sql
CREATE TABLE [embeddings] (
   [id] TEXT PRIMARY KEY,
   [embedding] BLOB
);
```
The same data can be provided as TSV data:
```
id    content
1     This is a test
2     This is another test
```
Then imported like this:
```bash
openai-to-sqlite embeddings embeddings.db data.tsv
```
Or as JSON data:
```json
[
  {"id": 1, "content": "This is a test"},
  {"id": 2, "content": "This is another test"}
]
```
Imported like this:
```
openai-to-sqlite embeddings embeddings.db data.json
```
In each of these cases the tool automatically detects the format of the data. It does this by inspecting the data itself - it does not consider the file extension.

If the automatic detection is not working, you can pass `--format json`, `csv` or `tsv` to explicitly specify a format:

```bash
openai-to-sqlite embeddings embeddings.db data.tsv --format tsv
```
### Importing data from standard input

You can use a filename of `-` to pipe data in to standard input:

```bash
cat data.tsv | openai-to-sqlite embeddings embeddings.db -
```

### Data from a SQL query

The `--sql` option can be used to read data to be embedded from the attached SQLite database. The query must return an `id` column and one or more text columns to be embedded.

```
openai-to-sqlite embeddings content.db \
  --sql "select id, title from documents"
```
This will create a `embeddings` table in the `content.db` database and populate it with embeddings calculated from the `title` column in that query.

You can also store embeddings in one database while reading data from another database, using the `--attach alias filename.db` option:

```
openai-to-sqlite embeddings embeddings.db \
  --attach documents documents.db \
  --sql "select id, title from documents.documents"
```
A progress bar will be displayed when using `--sql` that indicates how long the embeddings are likely to take to calculate.

The CSV/TSV/JSON options do not correctly display the progress bar. You can work around this by importing your data into SQLite first (e.g. [using sqlite-utils](https://sqlite-utils.datasette.io/en/stable/cli.html#inserting-json-data)) and then running the embeddings using `--sql`.

### Batching

Embeddings will be sent to the OpenAI embeddings API in batches of 100. If you know that your data is short strings you can increase the batch size, up to 2048, using the `--batch-size` option:

```bash
openai-to-sqlite embeddings embeddings.db data.csv --batch-size 2048
```

### Working with the stored embeddings

The `embedding` column is a SQLite blob containing 1536 floating point numbers encoded as a sequence of 4 byte values.

You can extract them back to an array of floating point values in Python like this:
```python
import struct

vector = struct.unpack(
    "f" * 1536, binary_embedding
)
```

### Searching embeddings with the search command

Having saved the embeddings for content, you can run searches using the `search` command:
```bash
openai-to-sqlite search embeddings.db 'this is my search term'
```
The output will be a list of cosine similarity scores and content IDs:
```
% openai-to-sqlite search blog.db 'cool datasette demo'
0.843 7849
0.830 8036
0.828 8195
0.826 8098
0.818 8086
0.817 8171
0.816 8121
0.815 7860
0.815 7872
0.814 8169
```

Add the `-t/--table` option if your embeddings are stored in a different table:
```bash
openai-to-sqlite search content.db 'this is my search term' -t documents
```

## Development

To contribute to this tool, first checkout the code. Then create a new virtual environment:

    cd openai-to-sqlite
    python -m venv venv
    source venv/bin/activate

Now install the dependencies and test dependencies:

    pip install -e '.[test]'

To run the tests:

    pytest
