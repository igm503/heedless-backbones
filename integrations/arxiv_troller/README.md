# Arxiv Troller

Discovery uses Arxiv Troller's JSON API at `/api/ingestion/`, maintained in the
[arxiv-troller](https://github.com/igm503/arxiv-troller) repository (see its README).
The client is `django/ingestion/sources.py` (`Troller`). It uses:

- `tag`: papers in the working tag;
- `search` with `type=tag`: the site's joint similarity search over the tag;
- `similar`: each tagged paper's 20 nearest papers;
- `copy_tag` and `bulk_add`: keep the working tag in sync with `backbones` and the
  papers already in the benchmark database.
