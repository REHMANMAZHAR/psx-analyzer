# PSX Analyzer — Web Deployment

## Easiest deployment: Streamlit Community Cloud

1. Create a free GitHub account at github.com if you do not already have one.
2. On GitHub, click **New repository**.
3. Name it `psx-analyzer` and choose **Public** for the simplest setup.
4. Click **Create repository**.
5. Click **uploading an existing file**.
6. Upload the complete contents of this folder, including:
   - `app.py`
   - `requirements.txt`
   - `.python-version`
   - the `.streamlit` folder containing `config.toml`
7. Click **Commit changes**.
8. Sign in at share.streamlit.io using GitHub.
9. Click **Create app**.
10. Select your `psx-analyzer` repository.
11. Set the main file path to `app.py`.
12. Choose an app URL and click **Deploy**.

Your analyzer will receive a web address ending in `.streamlit.app`.

## Alternative: Render

This package also contains `render.yaml`.

1. Sign in to Render with GitHub.
2. Create a new Blueprint or Web Service from the repository.
3. Render should detect `render.yaml` automatically.
4. Deploy the service.

Free hosting is suitable for testing. Free services may sleep when unused and can take time to wake up.

## Important market-data limitation

The app currently uses free Yahoo Finance daily data. This is not an official PSX real-time feed. Always check the displayed last candle date and compare prices with your broker terminal or the official PSX Data Portal before trading.
