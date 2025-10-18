# Magento Default Set Browser

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure secrets in `.streamlit/secrets.toml`:
   ```toml
   MAGENTO_BASE_URL = "https://example.com"
   MAGENTO_ADMIN_TOKEN = "****"
   ```

## Run locally

```bash
streamlit run app/streamlit_app.py
```

## Repository state

The `main` branch has been hard-reset to commit `5d6f692ada4f417250efa75e3950a2ad2a9c45f1` to remove all subsequent history and serve as the new baseline for future development.
