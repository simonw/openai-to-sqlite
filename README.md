# openai-to-sqlite

[![PyPI](https://img.shields.io/pypi/v/openai-to-sqlite.svg)](https://pypi.org/project/openai-to-sqlite/)
[![Changelog](https://img.shields.io/github/v/release/simonw/openai-to-sqlite?include_prereleases&label=changelog)](https://github.com/simonw/openai-to-sqlite/releases)
[![Tests](https://github.com/simonw/openai-to-sqlite/workflows/Test/badge.svg)](https://github.com/simonw/openai-to-sqlite/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/openai-to-sqlite/blob/master/LICENSE)

Save OpenAI API results to a SQLite database

**This tool is under active development**. It is not yet ready for production use.

## Installation

Install this tool using `pip`:

    pip install openai-to-sqlite

## Usage

For help, run:

    openai-to-sqlite --help

You can also use:

    python -m openai_to_sqlite --help

### Configuration

You will need an OpenAI API key to use this tool.

You can create one at https://beta.openai.com/account/api-keys

You can then either set the API key as an environment variable:

    export OPENAI_API_KEY=sk-...

Or pass it to each command using the `--token sk-...` option.

### Embeddings

The `embeddings` command can be used to calculate and store OpenID embeddings for strings of text.

Each embedding has a cost, so be sure to familiarize yourself with [the pricing](https://openai.com/api/pricing/) for the embedding model.

The command can accept data in four different ways:

- As a JSON file containing a list of objects
- As a CSV file
- As a TSV file
- Reading data from a SQLite database

For all of these formats there should be a `id` column, followed by one or more text columns.

The ID will be stored as the content ID. Any other columns will be concatenated together and used as the text to be embedded.

The embeddings from the API will then be saved as binary blobs in the `embeddings` table of the specified SQLite database - or another table, if you pass the `-t/--table` option.

#### CSV

Given a CSV file like this:

    id,content
    1,This is a test
    2,This is another test

Embeddings can be stored like so:

    openai-to-sqlite embeddings embeddings.db data.csv --csv

The `--csv` flag tells the tool that the input file is a CSV file. Without this it will attempt to guess the format.

The resulting schema looks like this:

```sql
CREATE TABLE [embeddings] (
   [id] TEXT PRIMARY KEY,
   [embedding] BLOB
);
```
The binary data can be extracted into a Python array of floating point numbers like this:
```python
import struct

vector = struct.unpack(
    "f" * 1536, binary_embedding
)
```

#### JSON

The expected JSON format looks like this:

```json
[
    {
        "id": "1",
        "content": "This is some text"
    },
    {
        "id": "2",
        "content": "This is some more text"
    }
]
```
This can be passed to the command like so:

    openai-to-sqlite embeddings embeddings.db data.json

Or piped to standard input (which works for CSV and TSV files too):

    cat data.json | openai-to-sqlite embeddings embeddings.db

#### Data from a SQL query

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

### Search

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
