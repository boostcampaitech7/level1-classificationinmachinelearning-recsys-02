import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf


class ApplyCCF:

  def __init__(self, data, y_var, x_vars, max_lag=120):
      self.data = data
      self.y_var = y_var
      self.x_vars = x_vars
      self.max_lag = max_lag


  def create_ccf_grid_plot(self):
      n_vars = len(self.x_vars)
      n_cols = 3
      n_rows = (n_vars + n_cols - 1) // n_cols

      fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
      fig.suptitle(f'Backward CCF Plots: {self.y_var} vs Other Variables (Last {self.max_lag} hours)', fontsize=16)

      for i, x_var in enumerate(self.x_vars):
          row = i // n_cols
          col = i % n_cols
          ax = axes[row, col] if n_rows > 1 else axes[col]

          y = self.data[self.y_var]
          x = self.data[x_var]
          
          # ccf value
          ccf_values = ccf(x[::-1], y[::-1], adjusted=False, nlags=self.max_lag)  # backward correlation 계산을 위해 [::-1]로 reverse
          lags = range(0, self.max_lag) 
      
          ax.bar(lags, ccf_values, width=0.3)
          ax.axhline(y=0, color='black', linestyle='-')
          ax.axhline(y=1.96/np.sqrt(self.max_lag), color='r', linestyle='--')
          ax.axhline(y=-1.96/np.sqrt(self.max_lag), color='r', linestyle='--')
          
          ax.set_title(f'{self.y_var} vs {x_var}')
          ax.set_xlabel('Lag (hours)')
          ax.set_ylabel('Cross-Correlation')
          ax.tick_params(axis='x', rotation=45)

      # Remove empty subplots
      for i in range(n_vars, n_rows * n_cols):
          row = i // n_cols
          col = i % n_cols
          fig.delaxes(axes[row, col] if n_rows > 1 else axes[col])

      plt.tight_layout()
      plt.subplots_adjust(top=0.95)
      return fig
