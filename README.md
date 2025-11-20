# 🌌 Futuristic Data Analysis Hub

A modern, modular data analysis application built with Streamlit, featuring a stunning cyberpunk-inspired UI and powerful data processing capabilities.

## ✨ Features

- 🗂 **Data Exploration** - Comprehensive data viewing and statistics
- 🔍 **Advanced Filtering** - Single and multi-condition filtering
- 📑 **Smart Selection** - Flexible row and column selection
- 🧹 **Data Cleaning** - Handle missing values, duplicates, and outliers
- 📈 **Rich Visualizations** - 7 different chart types with custom styling
- 📊 **Grouping & Aggregation** - Single and multi-column grouping
- 🔠 **Encoding** - One-hot and label encoding for categorical data
- 🤖 **Machine Learning** - Built-in model training and evaluation

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd data-analysis-app
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
data-analysis-app/
│
├── app.py                      # Main application entry point
├── requirements.txt            # Project dependencies
├── README.md                   # This file
│
├── styles/
│   └── theme.py               # Futuristic CSS theme
│
├── pages/
│   ├── __init__.py            # Package initializer
│   ├── explore.py             # Data exploration module
│   ├── filtering.py           # Data filtering module
│   ├── select.py              # Column/row selection module
│   ├── clean.py               # Data cleaning module
│   ├── visualize.py           # Data visualization module
│   ├── grouping.py            # Grouping & aggregation module
│   ├── encoding.py            # Encoding module
│   ├── modeling.py            # Machine learning module
│   └── about.py               # About page
│
└── utils/
    ├── __init__.py            # Package initializer
    ├── data_processor.py      # Data processing utilities
    └── ml_helper.py           # Machine learning utilities
```

## 🎨 Design Philosophy

This application features a **futuristic cyberpunk theme** with:
- 🌈 Animated gradient backgrounds
- ⚡ Neon glow effects (cyan & purple)
- 🎮 Smooth transitions and hover effects
- 🔮 Custom Orbitron & Rajdhani fonts
- 🌌 High-tech grid overlay

## 📚 Usage Guide

### 1. Upload Data
- Click "Browse files" or drag and drop a CSV file
- The app will display dataset metrics (rows, columns, size)

### 2. Explore
- Choose from various exploration options
- View head, tail, statistics, data types, and missing values

### 3. Clean
- Convert columns to numeric
- Fill missing values with mean, median, or mode
- Remove duplicates and unnecessary columns

### 4. Visualize
- Create bar charts, histograms, line plots
- Generate scatter plots and correlation heatmaps
- View box plots and pie charts

### 5. Filter & Select
- Apply single or multiple filters
- Select specific columns and rows
- Download filtered data

### 6. Group & Aggregate
- Group by single or multiple columns
- Apply aggregation functions (count, sum, mean, etc.)
- Create custom aggregations

### 7. Encode
- Apply one-hot encoding for nominal data
- Use label encoding for ordinal data
- View encoding mappings

### 8. Build Models
- Select features and target variable
- Choose from 3 model types
- Train and evaluate models
- View performance metrics and confusion matrix

## 🛠 Technical Stack

- **Streamlit** - Web framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical visualizations
- **Scikit-learn** - Machine learning

## 📦 Dependencies

```
streamlit==1.28.0
pandas==2.1.0
matplotlib==3.8.0
seaborn==0.13.0
scikit-learn==1.3.0
numpy==1.25.0
```

## 🎯 Key Modules

### Pages
Each page is a self-contained module with its own `render(df)` function:
- Clean separation of concerns
- Easy to test and maintain
- Simple to extend with new features

### Utils
Helper functions for common operations:
- **data_processor.py** - Data cleaning and transformation
- **ml_helper.py** - Machine learning model training and evaluation

### Styles
- **theme.py** - All CSS styling in one place

## 🔧 Customization

### Adding a New Page

1. Create a new file in `pages/` (e.g., `pages/newpage.py`)
2. Define a `render(df)` function
3. Import it in `pages/__init__.py`
4. Add it to the sidebar radio in `app.py`

```python
# pages/newpage.py
import streamlit as st
import pandas as pd

def render(df: pd.DataFrame):
    st.header("🆕 NEW PAGE")
    # Your code here
```

### Modifying the Theme

Edit `styles/theme.py` to change:
- Colors
- Fonts
- Animations
- Layout

## 👨‍💻 Developer

**Jatin Kumar**
- B.Tech (Electronics and Communication Engineering), 3rd Year
- Guru Nanak Dev University, Amritsar
- 📧 dhjatin4@gmail.com
- 📱 9888197119

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Built with ❤️ and ⚡
- Inspired by cyberpunk aesthetics
- Powered by the amazing Streamlit community

## 🐛 Bug Reports & Feature Requests

Feel free to open an issue or contact the developer directly.

## 🔮 Future Enhancements

- [ ] Support for more file formats (Excel, JSON, etc.)
- [ ] Advanced statistical tests
- [ ] More machine learning algorithms
- [ ] Data export in multiple formats
- [ ] User authentication
- [ ] Database integration
- [ ] Real-time data streaming

---

**Built with 🌌 for the future of data analysis**