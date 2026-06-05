"""
Nightly cron job — hits /refresh to update correlation data.
Run this from cron-job.org or any scheduler:
  URL: https://core.market-dashboard.com/refresh
  Schedule: Daily at 5:30 PM CT (23:30 UTC)

Or run locally:
  python cron_refresh.py
"""
import requests
import sys

URL = "https://core.market-dashboard.com/refresh"

try:
    r = requests.get(URL, timeout=30)
    print(f"Status {r.status_code}: {r.text}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
