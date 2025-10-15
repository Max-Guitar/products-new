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
