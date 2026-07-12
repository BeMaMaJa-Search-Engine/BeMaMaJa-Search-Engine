# Supabase Storage

The Streamlit app can load its runtime data from a private Supabase Storage
bucket named `search-data`.

- Organization: `MSE-project`
- Project: `mse-tuebingen-search`
- Region: `eu-central-1`
- Project URL: `https://xfifaguibdchyujbmace.supabase.co`
- Bucket: `search-data` (private, 50 MB per file)

The project URL is a public project identifier and is safe to keep in this
documentation. It does not grant access to the private bucket by itself.

## Stored files

- `index.json.gz`: downloaded once when Streamlit starts and cached in RAM.
- `raw_pages.json.gz`: loaded for full-page term matching and AI summaries,
  then cached in RAM.
- `preprocessed_pages.json`: remains a local build artifact and is not uploaded.

Local `data/index.json` and `data/raw_pages.json` files remain development
fallbacks if Supabase is unavailable.

## Streamlit secrets

Copy the placeholders from `.streamlit/secrets.toml.example` into the local
`.streamlit/secrets.toml` and into Streamlit Community Cloud secrets. Use a
server-side Supabase secret key, never a publishable key.

## Updating data

1. Rebuild the local JSON files.
2. Compress `index.json` and `raw_pages.json` with gzip.
3. Replace `index.json.gz` and `raw_pages.json.gz` in the private bucket.
4. Reboot the Streamlit app so its resource cache downloads the new version.

The current compressed files are below the Supabase Free upload limit. File
splitting is therefore not required.
