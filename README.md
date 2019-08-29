# pairs-pipeline

Pairs trading strategy that includes a research pipeline for identifying and selecting pairs. Tests all possible pairs in a universe for cointegration using the Johansen test, then runs in-sample backtests on all cointegrating pairs, then runs an out-of-sample backtest on the 5 best performing pairs. Calculates daily hedge ratios using the Johansen test and times entries and exits using Bollinger Bands. Trading rules adapted from Ernie Chan's book *Algorithmic Trading*. Runs in Moonshot on the universe of US ETFs.

## Clone in QuantRocket

CLI:

```shell
quantrocket codeload clone 'pairs-pipeline'
```

Python:

```python
from quantrocket.codeload import clone
clone("pairs-pipeline")
```

## Browse in GitHub

Start here: [pairs_pipeline/Introduction.ipynb](pairs_pipeline/Introduction.ipynb)

***

Find more code in QuantRocket's [Codeload Library](https://www.quantrocket.com/code/)
