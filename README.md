# Kelly Criterion Calculator 📈

A Python utility for computing the **Kelly fraction**, which determines the optimal proportion of capital to risk in trading or betting scenarios to maximize long-term growth.

This repository includes a simple and clear implementation of the **Kelly Criterion**, allowing you to input your win/loss statistics and receive the optimal bet sizing fraction.

---

## 🧮 What Is the Kelly Criterion?

The **Kelly Criterion** is a mathematical formula used to determine the ideal fraction of capital to allocate per trade (or bet) based on your historical performance. It balances **risk and growth** by maximizing expected logarithmic utility.

The formula is:

\[
f^* = W - \frac{1 - W}{R}
\]

where:

- **W** = Probability of a win (win rate)
- **R** = Ratio of average win to average loss

---

## 🚀 Features

- Computes the Kelly fraction given trade statistics  
- Handles invalid inputs (e.g., zero trades, missing losses, etc.)  
- Provides warnings for edge cases (like all winning or all losing trades)  
- Easily extendable for integration with financial data via `yfinance`

---

## 🧠 Example Usage

```python
from Kelly import calculate_kelly_fraction

# Example data
num_wins = 60
total_trades = 100
total_gain_from_wins = 12000.0
total_loss_from_losses = 8000.0

kelly_fraction = calculate_kelly_fraction(
    num_wins=num_wins,
    total_trades=total_trades,
    total_gain_from_wins=total_gain_from_wins,
    total_loss_from_losses=total_loss_from_losses
)

print(f"Optimal Kelly Fraction: {kelly_fraction:.2%}")
