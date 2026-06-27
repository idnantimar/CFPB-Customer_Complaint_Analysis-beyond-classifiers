# ============================================================
# API ref: https://cfpb.github.io/ccdb5-api/documentation/#/Complaints/get_
# ============================================================
import urllib.parse
API_Base = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
PARAMS = {
    "date_received_max": None, # '<=' filtering
    "date_received_min": None, # '>=' filtering
    "field": "all",
    "has_narrative": "true", # 95% data has only complaint tag and no text narrative, which we pre-filer at server
    "no_aggs": "true",
    "format": "csv"
}
make_url = lambda x,y : f"{API_Base}?{urllib.parse.urlencode(PARAMS|{"date_received_min":x,"date_received_max":y})}"
# ============================================================


# ============================================================
# Batch Extraction
# ============================================================
import time
from datetime import date
from dateutil.relativedelta import relativedelta
import urllib.request
import random

range_start = date(2025,12,1)
n_months = 6


for i in range(n_months) : # non-overlapping date ranges
    range_end = range_start + relativedelta(months=1) - relativedelta(days=1)

    url = make_url(range_start.isoformat(),range_end.isoformat())
    print(url)
    # You can directly pass these url in `pandas.read_csv(...)`
    # alternatively you can create a one-time extraction and reuse the local copy for downsteam tasks
    urllib.request.urlretrieve(url,f"./data_{range_start:%Y%m}.csv")

    range_start = range_end + relativedelta(days=1)
    time.sleep(random.randint(30,60)) # avoid HTTP Error 429
# ============================================================



