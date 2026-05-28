# FinSight Platform: Multi-Modal Market Surveillance & Portfolio Optimization Engine

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Deep%20Learning-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![Database](https://img.shields.io/badge/Big%20Data-DuckDB-fff000.svg)](https://duckdb.org/)
[![UI](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![Academic Core](https://img.shields.io/badge/Curriculum-MTech%20Capstone-red.svg)](#)

FinSight is an enterprise-grade Hybrid Intelligent System designed to forecast multi-asset market trajectories and optimize investment portfolio capital distribution under extreme volatility conditions. By joining streaming unstructured semantic signals with high-frequency quantitative market variables, the platform provides rigorous risk mitigation and tactical asset allocation models built on a completely optimized, local-first production stack.

---

## 🎓 Academic Pillars Mapping

This architecture fulfills the core technical requirements of an advanced postgraduate technical curriculum:

| Core Curriculum Pillar | Concrete Technical Implementation inside FinSight Repository |
| :--- | :--- |
| **1. Big Data Analytics** | Out-of-core vectorized feature extraction pipelines running over massive historical asset structures using vectorized relational in-memory **DuckDB** engines (`ingest.py`). |
| **2. Natural Language Processing** | Aspect-based financial text mining calculating multi-dimensional semantic sentiment vectors using the **Google Gemini API** with robust dynamic key remappers (`sentiment.py`). |
| **3. Deep Learning** | A sequential recurrent **PyTorch LSTM Neural Network** optimized using gradient descent to model rolling volatility sequences across target equities (`model.py`). |
| **4. Soft Computing** | An evolutionary heuristics layer driven by a **Deterministic Genetic Algorithm** utilizing the **DEAP Framework** for multi-objective optimization (`optimize.py`). |
| **5. Business Analytics** | An interactive operational executive cockpit built via **Streamlit** displaying Sharpe Ratios, Drawdown charts, and historical capital allocation weight matrices (`app.py`). |

---

## 🏗️ System Architecture & Data Pipeline Flow

The system is engineered as an asynchronous, decoupling-safe data pipeline where specialized workers transform raw operational inputs into low-variance predictive weights:

```text
                                  [ SYSTEM DATA FLOW PIPELINE ]
                                  
  +───────────────────────+        +─────────────────────────+        +───────────────────────+
  |  yfinance Ingestion   | ---->  |   DuckDB Feature Store  | ---->  |  PyTorch Time-Series  | ───┐
  |  (Structural Records) |        |   (Multi-Stage CTEs)    |        |      LSTM Network     |    |
  +───────────────────────+        +─────────────────────────+        +───────────────────────+    |
                                                │                                  │ (Volatility   |
                                                ▼                                  ▼  Forecasts)   |
  +───────────────────────+        +─────────────────────────+                                     ▼
  | Unstructured Text     | ---->  |  Google Gemini NLP API  | ───────────────────────────────> +─────────────+
  | Financial Headlines   |        |  (Sentiment Vectors)    |                                  | Evolutionary|
  +───────────────────────+        +─────────────────────────+                                  | DEAP Genetic|
                                                                                                |  Algorithm  |
                                                                                                +─────────────+
                                                                                                       │
                                                                                                       ▼
                                                                                                +─────────────+
                                                                                                |  Streamlit  |
                                                                                                | BI Cockpit  |
                                                                                                +─────────────+
```
## 🚀 Pipeline Breakdown & Architecture Core

The platform is engineered as a highly decoupled, asynchronous automated data pipeline where specialized scripts transform raw, noisy inputs into low-variance predictive assets:

### 1. 🗄️ Big Data Layer (`ingest.py`)
* **Ingestion & Processing:** Pulls high-volume market matrices asynchronously via the `yfinance` API. 
* **Vectorized Computations:** Rather than executing slow, row-by-row iteration loops, it shifts raw records straight into a vectorized in-memory **DuckDB** relational database instance.
* **Analytical CTE Alignment:** Utilizes multi-stage **Common Table Expressions (CTEs)** to cleanly isolate sequential lagging tracking windows, computing 20-day rolling sample volatility boundaries while completely preserving core `open`, `high`, and `low` price vectors for downstream visualization mechanics.

### 🧠 2. Semantic Extraction Layer (`sentiment.py`)
* **Natural Language Processing:** Processes global headline text arrays utilizing the **Google Gemini API SDK** to extract daily financial sentiment matrices per tracking ticker.
* **Defensive Feature Mapping:** Integrates a robust dictionary-level key remapper that intercepts structural variance (such as camelCase keys, raw text strings, or missing underscores) across local fallback caches.
* **Data Integration:** Normalizes incoming signals cleanly into uniform `avg_sentiment` and `sentiment_confidence` targets before outputting the processed `merged_features.parquet` matrix.

### 📉 3. Deep Sequential Layer (`model.py`)
* **Neural Architecture:** Implements a custom recurrent **PyTorch LSTM Neural Network** tracking hidden and cell state tensors across continuous historical lookback sequence windows.
* **Optimization Parameters:** Trained utilizing an **Adam Gradient Descent Optimizer** to map combined deep quantitative price features and text-mined semantic sentiment variables.
* **Predictive Yield:** Captures complex multi-dimensional temporal trends to forecast next-day forward-looking multi-asset market volatility vectors ($\sigma_p$).

### 🧬 4. Evolutionary Optimization Layer (`optimize.py`)
* **Soft Computing Engine:** Driven by the metaheuristic **DEAP Genetic Algorithm Framework** to manage asset risk weight allocations where traditional deterministic equations hit boundaries.
* **Heuristic Chromosomes:** Initializes candidate chromosome structure arrays representing target asset weights ($\omega$) across your tracking basket.
* **Generative Loops:** Runs iterative generational survival feedback loops (incorporating structured *Selection*, *Crossover*, and *Mutation* operations) to maximize localized Sharpe Ratio metrics calculated against the LSTM network's risk predictions.

### 🎨 5. Executive Presentation Layer (`app.py`)
* **Interactive UI Presentation:** A responsive, multi-page business intelligence cockpit built via **Streamlit** and **Plotly Vector Charting** libraries.
* **Defensive SDE Fault-Tolerance:** Integrates dynamic uppercase-to-lowercase string transformations and clone-on-missing schema structural fallbacks inside the caching loaders.
* **Operational Stability:** Ensures interactive candlestick timelines, moving average trends, and portfolio allocation pie charts render smoothly with **100% runtime dashboard uptime**.

## 🧮 Core Mathematical Formulations

The FinSight Platform shifts from classic empirical approximations to a tightly coupled, multi-stage mathematical engine. Below are the definitive formal models underpinning our data transformations, neural forecasting mechanics, and evolutionary optimizations.

---

### 1. 📊 Volatility & Log Returns Pipeline (`ingest.py`)

Raw equities asset streams are non-stationary time-series data. To stabilize variance and compute normalized statistical boundaries across historical intervals, the vectorized database engine converts raw close price matrices into continuous **Logarithmic Returns ($R_t$)**:

$$R_{t} = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

Where:
* $P_t$ = Normalized close price of the asset at trading interval $t$.
* $P_{t-1}$ = Normalized close price of the asset at trading interval $t-1$.

Using these log-stabilized distributions, the pipeline applies an out-of-core moving-window sample standard deviation to extract the **20-Day Rolling Volatility ($\sigma_{20d}$)**:

$$\sigma_{20d} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (R_i - \bar{R})^2}$$

Where:
* $N$ = Sliding lookback sequence horizon scale (fixed at $20$ active market sessions).
* $R_i$ = Logarithmic asset return computed at step $i$ inside the window context.
* $\bar{R}$ = Calculated arithmetic mean return over the current sliding sample segment.

---

### 🧠 2. Deep Sequential State Space Modeling (`model.py`)

The sequential neural architecture processes a combined tensor $X_t$ containing the quantitative market matrices mapped from DuckDB joined with the NLP semantic text-mined vectors ($\text{sent}_t$, $\text{conf}_t$) generated via the Google Gemini SDK:

$$X_t = \begin{bmatrix} \text{indicators}_t \\ \text{avg\_sentiment}_t \\ \text{sentiment\_confidence}_t \end{bmatrix}$$

This multi-modal input tensor flows into an itemized recurrent **Long Short-Term Memory (LSTM)** block, transitioning memory parameters over a continuous time space using standard gating mechanics:

$$\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, X_t] + b_f) && \text{(Forget Gate Layer)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, X_t] + b_i) && \text{(Input Gate Layer)} \\
\tilde{C}_t &= \tanh(W_c \cdot [h_{t-1}, X_t] + b_c) && \text{(Candidate Cell State Update)} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t && \text{(Final Cell Memory Update)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, X_t] + b_o) && \text{(Output Gate Layer)} \\
h_t &= o_t \odot \tanh(C_t) && \text{(Hidden Tensor Target Vector)}
\end{aligned}$$

Where:
* $W_f, W_i, W_c, W_o$ represent optimized weights tensors, and $b_f, b_i, b_c, b_o$ represent bias parameters.
* $h_t$ is the dynamic output hidden sequence state vector, used by the linear layer to generate the forward prediction vector $\hat{\sigma}_{t+1}$.
* $\sigma$ represents the standard Sigmoid gating function, and $\odot$ dictates elements-wise Hadamard tensor multiplication.

The loss function minimizing regression weight error across training iterations is mapped via **Mean Squared Error (MSE)** loss optimization:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{M} \sum_{k=1}^{M} \left( \sigma_{k, \text{actual}} - \hat{\sigma}_{k, \text{predicted}} \right)^2$$

---

### 🧬 3. Metaheuristic Evolutionary Optimization (`optimize.py`)

Rather than relying on classic quadratic matrix inversions that break under heavy non-linear sentiment variations, capital allocation weights are modeled as an evolutionary chromosome array vector $\omega$.

The metaheuristic **DEAP Genetic Algorithm Framework** maximizes a specialized multi-objective **Sharpe Ratio Fitness Function** ($\Phi$). The portfolio variance risk denominator ($\sigma_p$) is derived dynamically from the PyTorch LSTM predictive layer:

$$\text{Maximize } \Phi(\omega) = \frac{R_p - R_f}{\sigma_p}$$

The component values for individual portfolio expected return ($R_p$) and unified tracking portfolio variance risk ($\sigma_p$) are calculated as:

$$R_p = \sum_{j=1}^{K} \omega_j \cdot E(R_j)$$

$$\sigma_p = \sqrt{\sum_{j=1}^{K} \omega_j^2 \cdot \hat{\sigma}_j^2 + 2\sum_{j=1}^{K}\sum_{m > j}^{K} \omega_j \omega_m \cdot \text{Cov}(j, m)}$$

#### ⚠️ Boundary Constraints Management:
To ensure realistic, long-only real-world trading positions without short-selling or artificial leverage leaks, the chromosome mutations are tightly bound by the following conditions:

$$\sum_{j=1}^{K} \omega_j = 1.0 \quad \text{and} \quad \omega_j \ge 0 \quad \forall j \in \{1, 2, \dots, K\}$$

If an evolutionary crossover or mutation operation breaches these criteria, the pipeline passes the array into a normalizing vector function to restore compliance before fitness recalculation:

$$\omega_{j, \text{normalized}} = \frac{\max(0, \omega_j)}{\sum_{m=1}^{K} \max(0, \omega_m)}$$
