# Amazon Product Sales Analysis, Prediction, and Forecasting

A data analysis and forecasting project for Amazon product sales. It includes a Jupyter notebook for exploration and model development, plus a FastAPI web application with an interactive dashboard.

## Features

- Explore sales, revenue, profit, ratings, discounts, brands, categories, warehouses, and fulfilment types.
- Predict total units sold for a product using a trained Random Forest model.
- Forecast monthly units sold with a damped ETS model and show prediction intervals.
- View discount-band, high-return, and restocking insights.
- Access the same functionality through the web dashboard and JSON API.

## Project Structure

```text
Backend/       FastAPI application and analytics logic
Dataset/       Source and enriched Amazon product datasets
Frontend/      Dashboard HTML, CSS, and JavaScript
Notebook/      Analysis notebook and trained model artifact
requirements.txt
```

The application expects these files at their current locations:

- `Dataset/amazon_enriched.csv`
- `Notebook/rf_units_sold_model.joblib`

Do not move or rename these files unless you also update the paths in `Backend/analytics.py`.

## Requirements

- Python 3.10 or newer
- Packages listed in `requirements.txt`

## Setup on Windows

Open PowerShell in the project root and create a virtual environment:

```powershell
py -m venv myenv
```

Install the dependencies into that environment:

```powershell
.\myenv\Scripts\python.exe -m pip install --upgrade pip
.\myenv\Scripts\python.exe -m pip install -r requirements.txt
```

To use the environment in VS Code notebooks, choose the `myenv` Python interpreter/kernel. If it does not appear in the kernel list, install the notebook kernel package:

```powershell
.\myenv\Scripts\python.exe -m pip install ipykernel
```

## Run the Dashboard and API

From the project root:

```powershell
.\myenv\Scripts\python.exe -m uvicorn Backend.main:app --reload
```

Alternatively, from inside the `Backend` folder:

```powershell
..\myenv\Scripts\python.exe -m uvicorn main:app --reload
```

Open the dashboard at <http://127.0.0.1:8000>. Interactive API documentation is available at <http://127.0.0.1:8000/docs>. The dashboard loads Chart.js from a CDN, so an internet connection is required for its charts.

The first application startup loads the dataset, trained model, and forecast. Keep the server running while using the dashboard.

## API Routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Check that the API is running |
| `GET` | `/api/summary` | Return dashboard summary metrics |
| `GET` | `/api/eda` | Return category, brand, region, fulfilment, distribution, and product analysis |
| `GET` | `/api/options` | Return valid product input choices |
| `POST` | `/api/predict` | Predict units sold for the supplied product details |
| `GET` | `/api/forecast?months=6` | Return historical and forecast monthly units (1–24 months) |
| `GET` | `/api/insights?horizon=3` | Return discount, returns, and restocking insights |

## Run the Notebook

Open `Notebook/Notebook.ipynb` in VS Code or Jupyter and select the project `myenv` kernel. Run the notebook cells in order. The notebook reads `Dataset/amazon_enriched.csv` and trains and saves the model artifact used by the API at `Notebook/rf_units_sold_model.joblib`.

If a module import fails, install the project dependencies into the notebook's active kernel by running this in a notebook cell:

```python
%pip install -r ../requirements.txt
```

Then restart the kernel and rerun the affected cells. The relative requirements path assumes the notebook is running from the `Notebook` directory; if it is not, use the path to `requirements.txt` from the notebook's current working directory.

## Notes

- The forecast is based on monthly sales reconstructed by distributing each product's total units evenly over its recorded active sale months. Treat the resulting forecast as exploratory, not as a forecast from transaction-level sales history.
- The API allows cross-origin requests from any origin for development convenience. Restrict `allow_origins` in `Backend/main.py` before deploying it publicly.
