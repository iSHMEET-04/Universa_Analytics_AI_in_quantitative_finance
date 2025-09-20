# Universa_Analytics_AI_in_quantitative_finance
Deep learning for financial time series: LSTM and Transformer models predict multi-asset returns, powering robust adaptive asset allocation with full code, risk controls, and research-grade evaluation.

- Place raw historical data in `data/raw.csv`

## Pipeline

1. **Run preprocessing:**  
  `python ISH_preprocess.py`
2. **Train and generate predictions:**  
  `python ISH_modeling_lstm.py`  
  `python ISH_modeling_transformer.py`
3. **Evaluate portfolio:**  
  `python ISH_model_eval.py`  
  Results, charts, and metrics will appear in `results/` and standard output.

## Data
- **raw.csv**: Must contain daily price columns for each asset.

## Output
- **All model and evaluation scripts produce predictions, portfolio NAV trajectory, & key metrics**


