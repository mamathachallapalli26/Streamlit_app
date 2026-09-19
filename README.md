# Mental Health in Tech Survey EDA

An interactive Streamlit dashboard for exploring the 2014 OSMI Mental Health in Tech Survey. The app presents exploratory data analysis on mental health, workplace support, treatment, and related experiences in the technology industry.

## Features

- Interactive survey overview and key metrics
- Visual exploration of mental health and treatment responses
- Analysis of workplace support and attitudes toward mental health
- Plotly charts with responsive Streamlit controls
- Original exploratory analysis notebook included for reference

## Project Files

| File | Description |
| --- | --- |
| `app.py` | Streamlit dashboard application |
| `survey.csv` | OSMI Mental Health in Tech Survey dataset |
| `Mental_Health_in_Tech_Survey_EDA.ipynb` | Exploratory analysis notebook |
| `requirements.txt` | Python dependencies |

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/mamathachallapalli26/Streamlit_app.git
cd Streamlit_app
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` by default.

## Data Source

The dashboard uses the OSMI Mental Health in Tech Survey conducted in 2014. The dataset is included in this repository as `survey.csv` for local analysis.

## Author

Challapalli Mamatha