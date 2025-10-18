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

## Restore prompt 
Создай бэкап-ветку от текущего main: backup/main-$(date +%Y%m%d-%H%M).

От коммита 031a4fbf93a47f2c6af261dc8a0e08d4aa130596 создай ветку restore/031a4fbf.

Открой PR restore/031a4fbf → main с заголовком “Restore repository to 031a4fbf” и описанием “This PR restores the tree to commit 031a4fbf without history rewrite.”

Убедись, что PR полностью откатывает изменения после 031a4fbf (дифф пустой относительно того снимка), и влей PR (Merge).
