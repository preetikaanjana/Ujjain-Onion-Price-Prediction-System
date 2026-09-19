# Data Source Documentation: Ujjain Onion Mandi Prices (2024–2026)

## 1. Official Government Source & Provenance
- **Source:** Directorate of Marketing & Inspection (DMI), Ministry of Agriculture & Farmers Welfare, Government of India.
- **National Portal:** [AGMARKNET 2.0](https://agmarknet.gov.in/) & Open Government Data Platform ([data.gov.in](https://data.gov.in/))
- **Access / API Method:** Programmatic ingestion via official AGMARKNET 2.0 REST API:
  `https://api.agmarknet.gov.in/v1/prices-and-arrivals/date-wise/specific-commodity?year={year}&month={month}&stateId=19&commodityId=23`
- **Geographical Scope:** Madhya Pradesh (State ID: 19), District: Ujjain (District ID: 335)
- **Market (APMC):** `Ujjain APMC` (Market ID: 186 / 3053)
- **Commodity:** `Onion` (Commodity ID: 23)
- **Date Downloaded / Verified:** September 19, 2026

---

## 2. Dataset Metadata & Specification

| Attribute | Specification |
| :--- | :--- |
| **State** | Madhya Pradesh |
| **District** | Ujjain |
| **Market / Mandi** | Ujjain APMC |
| **Commodity** | Onion |
| **Total Records** | 1,086 market price records (699 unique daily trading sessions) |
| **2026 Records** | 289 market price records across 168 active trading days |
| **Date Range** | `2024-01-01` to `2026-09-19` |
| **Calendar Span** | 993 calendar days |
| **Active Trading Days** | 699 days |
| **Non-Trading Days** | 294 days (regular Sundays, gazetted holidays, mandi closures) |
| **Price Unit** | Indian Rupees per Quintal (₹ / 100 kg) |
| **Arrivals Unit** | Metric Tonnes |
| **Missing Values** | 0 null values across all fields |

---

## 3. Data Schema & Aggregation Rule

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `date` | Date (`YYYY-MM-DD`) | Date of the wholesale auction |
| `state` | String | State name (`Madhya Pradesh`) |
| `district` | String | Administrative district (`Ujjain`) |
| `market` | String | APMC market yard (`Ujjain APMC`) |
| `commodity` | String | Agricultural commodity (`Onion`) |
| `variety` | String | Commercial variety (`Onion`, `White`, `Small`, etc.) |
| `arrivals_tonnes` | Float | Quantity arrived at mandi yard in Metric Tonnes |
| `min_price` | Float | Minimum auction bid in ₹/Quintal |
| `max_price` | Float | Maximum auction bid in ₹/Quintal |
| `modal_price` | Float | **Primary Target:** Transaction price in ₹/Quintal |

### Aggregation Rule:
Aggregates multiple available market-price records for the same trading day into a daily price series using the implemented aggregation rule:
- `modal_price`: arithmetic mean across available daily price records.
- `min_price`: minimum across daily price records.
- `max_price`: maximum across daily price records.
- `arrivals_tonnes`: sum of reported arrival volume across records on that day.

---

## 4. Summary Statistics (2024–2026)

| Metric | Min Price (₹/Q) | Max Price (₹/Q) | Modal Price (₹/Q) [Target] | Arrivals (Tonnes) |
| :--- | :--- | :--- | :--- | :--- |
| **Mean** | ₹ 491.04 | ₹ 1,964.77 | ₹ 1,451.26 | 134.28 Tonnes |
| **Std Dev** | ₹ 584.86 | ₹ 1,116.87 | ₹ 1,099.79 | 148.91 Tonnes |
| **Minimum** | ₹ 80.00 | ₹ 249.00 | ₹ 100.00 | 0.05 Tonnes |
| **25%** | ₹ 200.00 | ₹ 1,180.00 | ₹ 600.00 | 25.40 Tonnes |
| **Median (50%)** | ₹ 300.00 | ₹ 1,500.00 | ₹ 1,120.07 | 78.50 Tonnes |
| **75%** | ₹ 520.47 | ₹ 2,486.55 | ₹ 2,097.50 | 192.15 Tonnes |
| **Maximum** | ₹ 4,041.00 | ₹ 11,831.00 | ₹ 5,000.00 | 1,420.00 Tonnes |

---

## 5. 2026 Specific Data Statistics (Jan 1, 2026 to Sep 19, 2026)
- **2026 Active Trading Days:** 168 active trading days (289 market price records)
- **2026 Date Span:** `2026-01-01` to `2026-09-19`
- **2026 Mean Modal Price:** ₹ 1,350.82 / Quintal
- **2026 Min Modal Price:** ₹ 150.00 / Quintal
- **2026 Max Modal Price:** ₹ 4,300.00 / Quintal
- **Latest 2026 Recorded Modal Price (Sep 19, 2026):** ₹ 3,549.96 / Quintal
