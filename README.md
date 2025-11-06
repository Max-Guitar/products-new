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

От коммита 716e0d9df3ecd3ca5347c50414515b33af7452ea создай ветку restore/716e0d9.

Открой PR restore/716e0d9 → main с заголовком “Restore repository to 716e0d9” и описанием “This PR restores the tree to commit 716e0d9 without history rewrite.”

Убедись, что PR полностью откатывает изменения после 716e0d9 (дифф пустой относительно того снимка), и влей PR (Merge).
