# `dream_go/database:0.5.0`

This image is responsible for storing the results produced by the other images
in a persistent manner. To do this it provides a _RESTful_ interface over HTTP
that can be easily accessed using command-line utilities, such as `curl`. The
API supports four kinds of queries and one `POST`:

- `[table]/recent/[n]` _(GET)_

  Gets the `n` most recent entries from the `table` collection.

- `[table]/recent/[n]/[field]` _(GET)_

  Gets the value of `field` from the `n` most recent entires from the `table`
  collection.

- `[table]/count/[field]` _(GET)_

  Gets the number of rows for each distinct value of `field` in the `table`
  collection.

- `[table]/[name]` _(GET)_

  Gets the record with the name `name` from the `table` collection.

- `[table]` _(POST)_

  Adds the content of the request as a new record in the `table` collection with
  a randomly generated name.

## Running

```bash
make docker
```
