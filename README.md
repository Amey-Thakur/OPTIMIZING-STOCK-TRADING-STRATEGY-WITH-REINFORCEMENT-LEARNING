

<div align="center">

  <a name="readme-top"></a>
  # Optimizing Stock Trading Strategy With Reinforcement Learning
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
  ![Status](https://img.shields.io/badge/Status-Completed-success)
  [![Technology](https://img.shields.io/badge/Technology-Python%20%7C%20Reinforcement%20Learning-blueviolet)](https://github.com/Amey-Thakur/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-REINFORCEMENT-LEARNING)
  [![Developed by Amey Thakur & Mega Satish](https://img.shields.io/badge/Developed%20by-Amey%20Thakur%20%26%20Mega%20Satish-blue.svg)](https://github.com/Amey-Thakur/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-REINFORCEMENT-LEARNING)

  A machine learning study demonstrating the application of **Reinforcement Learning (Q-Learning)** algorithms to optimize stock trading strategies and maximize portfolio returns.
  
  **[Source Code](https://github.com/Amey-Thakur/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-REINFORCEMENT-LEARNING/tree/main/Source%20Code)** &nbsp;·&nbsp; **[Kaggle Notebook](https://www.kaggle.com/ameythakur20/stock-price-prediction-model)** &nbsp;·&nbsp; **[Video Demo](https://youtu.be/Q82a93hjxJE)** &nbsp;·&nbsp; **[Live Demo](https://huggingface.co/spaces/ameythakur/Stock-Trading-RL)**

  <br>
  
  <a href="https://youtu.be/Q82a93hjxJE">
    <img src="https://img.youtube.com/vi/Q82a93hjxJE/maxresdefault.jpg" alt="Video Demo" width="70%">
  </a>

</div>

---

<div align="center">

  [Authors](#authors) &nbsp;·&nbsp; [Overview](#overview) &nbsp;·&nbsp; [Features](#features) &nbsp;·&nbsp; [Structure](#project-structure) &nbsp;·&nbsp; [Results](#results) &nbsp;·&nbsp; [Quick Start](#quick-start) &nbsp;·&nbsp; [Usage Guidelines](#usage-guidelines) &nbsp;·&nbsp; [License](#license) &nbsp;·&nbsp; [About](#about-this-repository) &nbsp;·&nbsp; [Acknowledgments](#acknowledgments)

</div>

---

<!-- AUTHORS -->
<div align="center">

  <a name="authors"></a>
  ## Authors

  | <a href="https://github.com/Amey-Thakur"><img src="https://github.com/Amey-Thakur.png" width="150" height="150" alt="Amey Thakur"></a><br>[**Amey Thakur**](https://github.com/Amey-Thakur)<br><br>[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--5644--1575-green.svg)](https://orcid.org/0000-0001-5644-1575) | <a href="https://github.com/msatmod"><img src="Mega/Mega.png" width="150" height="150" alt="Mega Satish"></a><br>[**Mega Satish**](https://github.com/msatmod)<br><br>[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--1844--9557-green.svg)](https://orcid.org/0000-0002-1844-9557) |
  | :---: | :---: |

</div>


> [!IMPORTANT]
> ### 🤝🏻 Special Acknowledgement
> *Special thanks to **[Mega Satish](https://github.com/msatmod)** for her meaningful contributions, guidance, and support that helped shape this work.*

---

<!-- OVERVIEW -->
<a name="overview"></a>
## Overview

**Optimizing Stock Trading Strategy With Reinforcement Learning** is a Data Science study conducted as part of the **Internship** at **Technocolabs Software**. The project focuses on the development of an intelligent agent capable of making autonomous trading decisions (Buy, Sell, Hold) to maximize profitability.

By leveraging **Q-Learning**, the system models the market environment where an agent learns optimal strategies based on price movements and moving average crossovers. The model is visualized via a **Streamlit** web application for real-time strategy simulation.

### Computational Objectives
The analysis is governed by strict **exploratory and modeling principles** ensuring algorithmic validity:
*   **State Representation**: utilization of Short-term and Long-term Moving Average crossovers to define market states.
*   **Action Space**: Discrete action set (Buy, Sell, Hold) optimized through reward feedback.
*   **Policy Optimization**: Implementing an Epsilon-Greedy strategy to balance exploration and exploitation of trading rules.

> [!NOTE]
> ### Research Impact
> This project was published as a research paper and successfully demonstrated the viability of RL agents in simulated trading environments. The work received official recognition from Technocolabs Software including an **Internship Completion Certificate** and **Letter of Recommendation**.
>
> *   [ResearchGate](http://dx.doi.org/10.13140/RG.2.2.13054.05440)
> *   [Project Completion Letter](https://github.com/Amey-Thakur/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-REINFORCEMENT-LEARNING/blob/main/Technocolabs/Technocolabs%20Software%20-%20Data%20Scientist%20-%20Project%20Completion%20Letter.pdf)

### Resources

| # | Resource | Description | Date |
| :---: | :--- | :--- | :--- |
| 1 | [**Source Code**](https://github.com/Amey-Thakur/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-REINFORCEMENT-LEARNING/tree/main/Source%20Code) | Complete production repository and scripts | — |
| 2 | [**Kaggle Notebook**](https://www.kaggle.com/ameythakur20/stock-price-prediction-model) | Interactive Jupyter notebook for model training | — |
| 3 | [**Dataset**](https://github.com/Amey-Thakur/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-REINFORCEMENT-LEARNING/blob/main/Source%20Code/all_stocks_5yr.csv) | Historical stock market data (5 Years) | — |
| 4 | [**Technical Specification**](docs/SPECIFICATION.md) | System architecture and specifications | — |
| 5 | [**Technical Report**](Technocolabs/PROJECT%20REPORT.pdf) | Comprehensive archival project documentation | September 2021 |
| 6 | [**Blueprint**](https://github.com/Amey-Thakur/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-REINFORCEMENT-LEARNING/blob/main/Technocolabs/AMEY%20THAKUR%20-%20BLUEPRINT.pdf) | Initial project design and architecture blueprint | September 2021 |


> [!TIP]
> ### Market Adaptation
> The Q-Learning agent's performance relies heavily on the quality of historical data. Regular retraining with recent market data is recommended to adapt the Q-Table's values to shifting market trends and volatility patterns.

---

<!-- FEATURES -->
<a name="features"></a>
## Features

| Component | Technical Description |
|-----------|-----------------------|
| **Data Ingestion** | Automated loading and processing of historical stock data (CSV). |
| **Trend Analysis** | Computation of 5-day and 1-day Moving Averages to identify trend signals. |
| **RL Agent** | **Q-Learning** implementation with state-action mapping for decision autonomy. |
| **Portfolio Logic** | Dynamic tracking of cash, stock holdings, and total net worth over time. |
| **Visualization** | Interactive **Streamlit** dashboard using **Plotly** for financial charting. |

> [!NOTE]
> ### Empirical Context
> Stock markets are stochastic environments. This project simplifies the state space to Moving Average crossovers to demonstrate the foundational capabilities of Reinforcement Learning in financial contexts, prioritizing pedagogical clarity over high-frequency trading complexity.

### Tech Stack
-   **Runtime**: Python 3.x
-   **Machine Learning**: NumPy, Pandas
-   **Visualization**: Streamlit, Plotly, Matplotlib, Seaborn
-   **Algorithm**: Q-Learning (Reinforcement Learning)

---

<!-- STRUCTURE -->
<a name="project-structure"></a>
## Project Structure

```python
OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-REINFORCEMENT-LEARNING/
│
├── docs/                                            # Technical Documentation
│   └── SPECIFICATION.md                             # Architecture & Design Specification
│
├── Mega/                                            # Archival Attribution Assets
│   ├── Filly.jpg                                    # Companion (Filly)
│   ├── Mega.png                                     # Author Profile Image (Mega Satish)
│   └── ...                                          # Additional Attribution Files
│
├── screenshots/                                     # Application Screenshots
│   ├── 01-landing-page.png                          # Home Interface
│   ├── 02-amzn-trend.png                            # Stock Trend Visualization
│   ├── 03-portfolio-growth.png                      # Portfolio Value Over Time
│   └── 04-alb-trend.png                             # Analysis Example
│
├── Source Code/                                     # Core Implementation
│   ├── Train_model/                                 # Training Notebooks
│   │   └── Model.ipynb                              # Q-Learning Implementation
│   │
│   ├── .streamlit/                                  # Streamlit Configuration
│   ├── all_stocks_5yr.csv                           # Historical Dataset
│   ├── model.pkl                                    # Trained Q-Table (Pickle)
│   ├── Procfile                                     # Heroku Deployment Config
│   ├── requirements.txt                             # Dependencies
│   ├── setup.sh                                     # Environment Setup Script
│   └── Stock-RL.py                                  # Main Application Script
│
├── Technocolabs/                                    # Internship Artifacts
│   ├── AMEY THAKUR - BLUEPRINT.pdf                  # Design Blueprint
│   ├── Optimizing Stock Trading...pdf               # Research Paper
│   ├── PROJECT REPORT.pdf                           # Final Project Report
│   └── ...                                          # Internship Completion Documents
│
├── .gitattributes                                   # Git configuration
├── .gitignore                                       # Repository Filters
├── CITATION.cff                                     # Scholarly Citation Metadata
├── codemeta.json                                    # Machine-Readable Project Metadata
├── LICENSE                                          # MIT License Terms
├── README.md                                        # Project Documentation
└── SECURITY.md                                      # Security Policy
```

---

<!-- RESULTS -->
<a name="results"></a>
## Results

<div align="center">

  <b>1. User Interface: Landing Page</b>
  <br>
  <i>The Streamlit-based dashboard allows users to select stocks and define investment parameters for real-time strategy optimization.</i>
  <br><br>
  <img src="screenshots/01-landing-page.png" alt="Landing Page" width="80%">

  <br><br>

  <b>2. Market Analysis: Stock Trend</b>
  <br>
  <i>Historical price visualization identifying long-term upward trends suitable for momentum-based trading strategies.</i>
  <br><br>
  <img src="screenshots/02-amzn-trend.png" alt="Stock Trend" width="80%">

  <br><br>

  <b>3. Strategy Evaluation: Portfolio Growth</b>
  <br>
  <i>Simulation of portfolio value over time, demonstrating the cumulative return generated by the agent against the initial capital.</i>
  <br><br>
  <img src="screenshots/03-portfolio-growth.png" alt="Portfolio Growth" width="80%">

  <br><br>

  <b>4. Risk Assessment: Volatility Analysis</b>
  <br>
  <i>Trend analysis highlighting periods of high volatility where the agent adjusts exposure to mitigate risk.</i>
  <br><br>
  <img src="screenshots/04-alb-trend.png" alt="Volatility Analysis" width="80%">

</div>

---

<!-- QUICK START -->
<a name="quick-start"></a>
## Quick Start

### 1. Prerequisites
-   **Python 3.7+**: Required for runtime execution. [Download Python](https://www.python.org/downloads/)
-   **Streamlit**: For running the web application locally.

> [!WARNING]
> **Data Consistency**
>
> The Q-Learning agent depends on proper state definitions. Ensure that the input dataset contains the required `date`, `close`, and `Name` columns to correctly compute the Moving Average crossovers used for state discretization.

### 2. Installation
Establish the local environment by cloning the repository and installing the computational stack:

```bash
# Clone the repository
git clone https://github.com/Amey-Thakur/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-REINFORCEMENT-LEARNING.git
cd OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-REINFORCEMENT-LEARNING

# Navigate to Source Code directory
cd "Source Code"

# Install dependencies
pip install -r requirements.txt
```

### 3. Execution
Launch the web server to start the prediction application:
```bash
streamlit run Stock-RL.py
```
**Access**: `http://localhost:8501/`

---

<!-- USAGE GUIDELINES -->
<a name="usage-guidelines"></a>
## Usage Guidelines

This repository is openly shared to support learning and knowledge exchange across the machine learning and algorithmic trading community.

**For Students**  
Use this project as reference material for understanding **Deep Reinforcement Learning (Q-Learning)**, **state-action discretisation**, and **financial reward shaping**. The source code is available for study to facilitate self-paced learning and exploration of **moving average strategies**.

**For Educators**  
This project may serve as a practical lab example or supplementary teaching resource for **Computational Finance**, **Artificial Intelligence**, and **Quantitative Trading** courses. Attribution is appreciated when utilizing content.

**For Researchers**  
The documentation and architectural approach may provide insights into **simplified market modeling**, **policy iteration in volatile environments**, and **industrial internship artifacts**.

---

<!-- LICENSE -->
<a name="license"></a>
## License

This academic submission, developed for the **Data Science Internship** at **Technocolabs Software**, is made available under the **MIT License**. See the [LICENSE](LICENSE) file for complete terms.

> [!NOTE]
> **Summary**: You are free to share and adapt this content for any purpose, even commercially, as long as you provide appropriate attribution to the original authors.

**Copyright © 2021 Amey Thakur & Mega Satish**

---

<!-- ABOUT -->
<a name="about-this-repository"></a>
## About This Repository

**Created & Maintained by**: [Amey Thakur](https://github.com/Amey-Thakur) & [Mega Satish](https://github.com/msatmod)  
**Role**: Data Science Interns  
**Program**: Data Science Internship  
**Organization**: [Technocolabs Software](https://technocolabs.com/)

This project features **Optimizing Stock Trading Strategy With Reinforcement Learning**, a study conducted as part of an industrial internship. It explores the practical application of Q-Learning in financial economics.

**Connect:** [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [LinkedIn](https://www.linkedin.com/in/amey-thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

### Acknowledgments

Grateful acknowledgment to [**Mega Satish**](https://github.com/msatmod) for her exceptional collaboration and scholarly partnership during the execution of this data science internship task. Her analytical precision, deep understanding of statistical modeling, and constant support were instrumental in refining the learning algorithms used in this study. Working alongside her was a transformative experience; her thoughtful approach to problem-solving and steady encouragement turned complex challenges into meaningful learning moments. This work reflects the growth and insights gained from our side-by-side academic journey. Thank you, Mega, for everything you shared and taught along the way.

Special thanks to the **mentors at Technocolabs Software** for providing this platform for rapid skill development and industrial exposure.

---

<div align="center">

  [↑ Back to Top](#readme-top)

  [Authors](#authors) &nbsp;·&nbsp; [Overview](#overview) &nbsp;·&nbsp; [Features](#features) &nbsp;·&nbsp; [Structure](#project-structure) &nbsp;·&nbsp; [Results](#results) &nbsp;·&nbsp; [Quick Start](#quick-start) &nbsp;·&nbsp; [Usage Guidelines](#usage-guidelines) &nbsp;·&nbsp; [License](#license) &nbsp;·&nbsp; [About](#about-this-repository) &nbsp;·&nbsp; [Acknowledgments](#acknowledgments)

  <br>

  📈 **[OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-REINFORCEMENT-LEARNING](https://huggingface.co/spaces/ameythakur/Stock-Trading-RL)**

  ---

  ### Presented as part of the Data Science Internship @ Technocolabs Software

  ---

  ### 🎓 [Computer Engineering Repository](https://github.com/Amey-Thakur/COMPUTER-ENGINEERING)

  **Computer Engineering (B.E.) - University of Mumbai**

  *Semester-wise curriculum, laboratories, projects, and academic notes.*

</div>
