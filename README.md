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

The first command supported by this tool is `embeddings`:

    openai-to-sqlite embeddings --help

This command can be fed a CSV (or JSON or TSV) file full of content, and it will use the 
OpenAI API to generate embeddings for each row.

The first column of the CSV file will be treated as the content ID. Any other columns will be concatenated together and used as the text to be embedded.

These embeddings will then be saved as binary blobs in the `embeddings` table of a 
SQLite database.

Given a CSV file like this:

    id,content
    1,This is a test
    2,This is another test

Embeddings can be stored like so:

    openai-to-sqlite embeddings embeddings.db data.csv --csv

The `--csv` flag tells the tool that the input file is a CSV file. Without this it
will attempt to guess.

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
### Search

Having saved the embeddings for content, you can run searches using the `search` command:

    openai-to-sqlite search embeddings.db 'this is my search term'

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
## Development

To contribute to this tool, first checkout the code. Then create a new virtual environment:

    cd openai-to-sqlite
    python -m venv venv
    source venv/bin/activate

Now install the dependencies and test dependencies:

    pip install -e '.[test]'

To run the tests:

    pytest
