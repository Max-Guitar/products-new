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

От коммита 9d2302860e1f616132117331174de4c0ef5fd54f создай ветку restore/c888ffc.

Открой PR restore/9d2302 → main с заголовком “Restore repository to 9d2302” и описанием “This PR restores the tree to commit 9d2302 without history rewrite.”

Убедись, что PR полностью откатывает изменения после 9d2302 (дифф пустой относительно того снимка), и влей PR (Merge).
