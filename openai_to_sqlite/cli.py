import click
import httpx
import json
import openai
from sqlite_utils.utils import rows_from_file, Format
from sqlite_utils.utils import sqlite3
import sqlite_utils
import struct
import tiktoken

sqlite3.enable_callback_tracebacks(True)

PRICING = {
    # model, (prompt, completion)
    # prices are per 1K tokens in 100ths of a cent
    "gpt4": (300, 600),
    "chatgpt": (20, 20),
    "ada": (4, 4),
    "babbage": (5, 5),
    "curie": (20, 20),
    "davinci": (200, 200),
}


@click.group()
@click.version_option()
def cli():
    "Tool for saving OpenAI API results to a SQLite database"


@cli.command()
@click.argument(
    "db_path",
    type=click.Path(file_okay=True, dir_okay=False, allow_dash=False),
)
@click.argument(
    "input_path",
    type=click.File("rb"),
    required=False,
)
@click.option(
    "--token",
    help="OpenAI API key",
    envvar="OPENAI_API_KEY",
)
@click.option(
    "table_name",
    "-t",
    "--table",
    default="embeddings",
    help="Name of the table to store embeddings in",
)
@click.option(
    "--format",
    type=click.Choice(["json", "csv", "tsv"]),
)
@click.option("--sql", help="Read input using this SQL query")
@click.option(
    "--attach",
    type=(str, click.Path(file_okay=True, dir_okay=False, allow_dash=False)),
    multiple=True,
    help="Additional databases to attach - specify alias and file path",
)
@click.option(
    "--batch-size",
    type=click.IntRange(1, 2048),
    default=100,
    help="Number of rows to send to OpenAI at once. Defaults to 100 - use a higher value if your text is smaller strings.",
)
def embeddings(db_path, input_path, token, table_name, format, sql, attach, batch_size):
    """
    Store embeddings for one or more text documents

    Input can be CSV, TSV or a JSON list of objects.

    The first column is treated as an ID - all other columns
    are assumed to be text that should be concatenated together
    in order to calculate the embeddings.
    """
    if not input_path and not sql:
        raise click.UsageError("Either --sql or input path is required")
    if not token:
        raise click.ClickException(
            "OpenAI API token is required, use --token=x or set the "
            "OPENAI_API_KEY environment variable"
        )
    db = sqlite_utils.Database(db_path)
    for alias, attach_path in attach:
        db.attach(alias, attach_path)
    table = db[table_name]
    if not table.exists():
        table.create(
            {"id": str, "embedding": bytes},
            pk="id",
        )
    expected_length = None
    if sql:
        rows = db.query(sql)
        count_sql = "select count(*) as c from ({})".format(sql)
        expected_length = next(db.query(count_sql))["c"]
    else:
        # Auto-detect
        try:
            rows, _ = rows_from_file(
                input_path, Format[format.upper()] if format else None
            )
        except json.JSONDecodeError as ex:
            raise click.ClickException(str(ex))
    # Use a click progressbar
    total_tokens = 0
    skipped = 0
    with click.progressbar(
        rows, label="Fetching embeddings", show_percent=True, length=expected_length
    ) as rows:
        # Run this batch_size at a time
        for batch in batch_rows(rows, batch_size):
            text_to_send = []
            ids_in_batch = []
            for row in batch:
                values = list(row.values())
                id = values[0]
                try:
                    table.get(id)
                    skipped += 1
                    continue
                except sqlite_utils.db.NotFoundError:
                    pass
                text = " ".join(v or "" for v in values[1:])
                ids_in_batch.append(id)
                # Actual limit is 8191 but we are being a bit short for safety:
                text_to_send.append(truncate_tokens(text, 8100))
            if not text_to_send:
                # Skip logic could have resulted in an empty batch
                continue
            # Send to OpenAI, but only if batch is populated - since
            # the skip logic could have resulted in an empty batch
            if text_to_send:
                response = httpx.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json={"input": text_to_send, "model": "text-embedding-ada-002"},
                )
                if response.status_code == 400:
                    click.echo(response.json()["error"], err=True)
                    click.echo(f"For IDs: {ids_in_batch} - skipping", err=True)
                    continue
                response.raise_for_status()
                data = response.json()
                total_tokens += data["usage"]["total_tokens"]
                results = data["data"]
                # Each one has an "embedding" and an "index"
                for result in results:
                    embedding = encode(result["embedding"])
                    table.insert(
                        {"id": ids_in_batch[result["index"]], "embedding": embedding},
                        replace=True,
                    )
    click.echo(f"Total tokens used: {total_tokens}", err=True)
    if skipped:
        click.echo(f"Skipped {skipped} rows that already existed", err=True)


@cli.command()
@click.argument(
    "db_path",
    type=click.Path(file_okay=True, dir_okay=False, allow_dash=False),
)
@click.argument("query")
@click.option(
    "--token",
    help="OpenAI API key",
    envvar="OPENAI_API_KEY",
)
@click.option(
    "table_name",
    "-t",
    "--table",
    default="embeddings",
    help="Name of the table containing the embeddings",
)
@click.option(
    "--count",
    type=int,
    default=10,
    help="Number of results to return",
)
def search(db_path, query, token, table_name, count):
    """
    Search embeddings using cosine similarity against a query.

    The query you pass will be embedded using the OpenAI API,
    then the closest matching records from the table will be shown.
    """
    if not token:
        raise click.ClickException(
            "OpenAI API token is required, use --token=x or set the "
            "OPENAI_API_KEY environment variable"
        )
    db = sqlite_utils.Database(db_path)
    table = db[table_name]
    if not table.exists():
        raise click.ClickException(f"Table {table_name} does not exist")
    # Fetch the embedding for the query
    response = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={"input": query, "model": "text-embedding-ada-002"},
    )
    response.raise_for_status()
    data = response.json()
    vector = data["data"][0]["embedding"]
    # Now calculate cosine similarity with everything in the database table
    other_vectors = [(row["id"], decode(row["embedding"])) for row in table.rows]
    results = [
        (id, cosine_similarity(vector, other_vector))
        for id, other_vector in other_vectors
    ]
    results.sort(key=lambda r: r[1], reverse=True)
    for id, score in results[:count]:
        print(f"{score:.3f} {id}")


@cli.command()
@click.argument(
    "db_path",
    type=click.Path(file_okay=True, dir_okay=False, allow_dash=False),
)
@click.argument("sql")
@click.option(
    "--token",
    help="OpenAI API key",
    envvar="OPENAI_API_KEY",
)
def query(db_path, sql, token):
    """
    Execute SQL query against the database, with access to these functions:

    \b
    - chatgpt(prompt) - run GPT3.5 against the prompt
    - chatgpt(prompt, system_prompt) - GPT 3.5 with a system prompt
    """
    if not token:
        raise click.ClickException(
            "OpenAI API token is required, use --token=x or set the "
            "OPENAI_API_KEY environment variable"
        )
    openai.api_key = token
    db = sqlite_utils.Database(db_path)

    used_tokens = []

    def usage(model, usage):
        assert usage.total_tokens == usage.completion_tokens + usage.prompt_tokens
        used_tokens.append((model, (usage.completion_tokens, usage.prompt_tokens)))

    # First pass to count executions
    todo_count = 0

    @db.register_function(name="chatgpt")
    def _(prompt):
        nonlocal todo_count
        todo_count += 1
        return ""

    @db.register_function(name="chatgpt")
    def _(prompt, system_prompt):
        nonlocal todo_count
        todo_count += 1
        return ""

    # Run it in a transaction and then roll it back
    with db.conn:
        db.execute(sql)
        db.conn.rollback()

    with click.progressbar(length=todo_count, label="Running", show_pos=True) as bar:
        # Register the functions to do the work
        @db.register_function(name="chatgpt", replace=True)
        def _(prompt):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            usage("chatgpt", response.usage)
            bar.update(1)
            return response.choices[0].message.content

        @db.register_function(name="chatgpt", replace=True)
        def _(prompt, system_prompt):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            usage("chatgpt", response.usage)
            bar.update(1)
            return response.choices[0].message.content

        with db.conn:
            cursor = db.execute(sql)
            if cursor.description:
                headers = [col[0] for col in cursor.description]
            click.echo("")
            for row in cursor:
                if headers:
                    row = dict(zip(headers, row))
                    click.echo(json.dumps(row))

    # Calculate price
    price_100th_cents = 0
    per_model = {}
    for model, (completion, prompt) in used_tokens:
        prices = PRICING[model]
        model_price = ((prompt * prices[0]) / 1000.0) + (
            (completion * prices[1]) / 1000.0
        )
        if model not in per_model:
            per_model[model] = {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "price_100th_cents": 0,
            }
        per_model[model]["completion_tokens"] += completion
        per_model[model]["prompt_tokens"] += prompt
        per_model[model]["price_100th_cents"] += model_price
        price_100th_cents += model_price
    cents = price_100th_cents / 100
    message = f"Total price: ${cents / 100:.4f}"
    if cents < 100:
        message += f" ({cents:.4f} cents)"
    click.echo(message, err=True)
    click.echo(json.dumps(round_floats(per_model), indent=4), err=True)


def round_floats(o):
    if isinstance(o, float):
        return round(o, 5)
    if isinstance(o, dict):
        return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x) for x in o]
    return o


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
    return dot_product / (magnitude_a * magnitude_b)


def decode(blob):
    return struct.unpack("f" * 1536, blob)


def encode(values):
    return struct.pack("f" * 1536, *values)


def batch_rows(rows, batch_size):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@cli.command()
@click.argument(
    "db_path",
    type=click.Path(file_okay=True, dir_okay=False, allow_dash=False),
)
@click.argument("entries", nargs=-1)
@click.option(
    "table_name",
    "-t",
    "--table",
    default="embeddings",
    help="Name of the table containing the embeddings",
)
@click.option(
    "--count",
    type=int,
    default=10,
    help="Number of results to return",
)
@click.option(
    "--all",
    is_flag=True,
    help="Calculate similar records for every record in the database",
)
@click.option(
    "--save", is_flag=True, help="Save the results to a table called similarities"
)
@click.option(
    "--save-table",
    default="similarities",
    help="Name of the table to save results to",
)
@click.option(
    "--recalculate-for-matches",
    is_flag=True,
    help="Recalculate the similarities for any that match the first set",
)
@click.option(
    "print_",
    "--print",
    is_flag=True,
    help="Echo the similarities even while saving them to the database",
)
def similar(
    db_path,
    entries,
    table_name,
    count,
    all,
    save,
    save_table,
    recalculate_for_matches,
    print_,
):
    """
    Display similar entries to the entries provided.
    """
    db = sqlite_utils.Database(db_path)
    table = db[table_name]
    if not table.exists():
        raise click.ClickException(f"Table {table_name} does not exist")
    if not all and not entries:
        raise click.ClickException("Must specify entries or --all")
    if all:
        if entries:
            raise click.ClickException("Cannot specify entries when using --all")
        if recalculate_for_matches:
            raise click.ClickException(
                "Cannot use --recalculate-for-matches with --all"
            )
        entries = [row["id"] for row in table.rows]

    # We run two rounds - the first is for the things that were specified
    for round in (1, 2):
        next_round = []
        for entry in entries:
            try:
                row = table.get(entry)
            except sqlite_utils.db.NotFoundError:
                raise click.ClickException(f"Entry not found:" + entry)
            vector = decode(row["embedding"])
            # Now calculate cosine similarity with everything in the database table
            other_vectors = [
                (row["id"], decode(row["embedding"])) for row in table.rows
            ]
            results = [
                (id, cosine_similarity(vector, other_vector))
                for id, other_vector in other_vectors
            ]
            results.sort(key=lambda r: r[1], reverse=True)
            if print_ or (not save):
                click.echo(results[0][0])
            top_results = results[1 : count + 1]
            for id, score in top_results:
                if print_ or (not save):
                    click.echo(f"  {score:.3f} {id}")
                next_round.append(id)
            if save:
                db[save_table].insert_all(
                    [
                        {
                            "id": entry,
                            "other_id": id,
                            "score": score,
                        }
                        for id, score in top_results
                    ],
                    pk=("id", "other_id"),
                    replace=True,
                )
        if not recalculate_for_matches or not next_round:
            break
        else:
            entries = next_round


encoding = None


def count_tokens(string: str) -> int:
    """Returns the number of tokens in a text string."""
    global encoding
    if encoding is None:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))


def truncate_tokens(text: str, truncate: int) -> str:
    global encoding
    if encoding is None:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    tokens = tokens[:truncate]
    return encoding.decode(tokens)
