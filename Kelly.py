import yfinance as yf
import pandas as pd
import numpy as np 

def calculate_kelly_fraction(
    num_wins: int,
    total_trades: int,
    total_gain_from_wins: float,
    total_loss_from_losses: float
) -> float:

    if total_trades <= 0:
        print("Error: Total trades must be greater than zero for Kelly calculation.")
        return 0.0

    # Calculate Win Probability (W)
    win_probability = num_wins / total_trades

    # Calculate Win/Loss Ratio (R)
    # R = (Average Gain of Winning Trades) / (Average Loss of Losing Trades)
    num_losses = total_trades - num_wins

    if num_losses > 0 and total_loss_from_losses <= 0:
        print("Error: Total loss from losses must be positive if there are losing trades.")
        return 0.0
    if num_wins > 0 and total_gain_from_wins <= 0:
        print("Error: Total gain from wins must be positive if there are winning trades.")
        return 0.0

    avg_gain = total_gain_from_wins / num_wins if num_wins > 0 else 0.0
    avg_loss = total_loss_from_losses / num_losses if num_losses > 0 else 0.0

    if avg_loss == 0:
        # If no losses, and there are wins, R is effectively infinite.
        # This implies a very high Kelly fraction, potentially 1.0.
        # In a real scenario, this is highly improbable over many trades.
        if num_wins > 0:
            print("Warning: No average loss from losing trades (all wins or zero losses). Assuming infinite R for Kelly calculation.")
            # For the formula f* = W - (1-W)/R, if R approaches infinity, (1-W)/R approaches 0.
            # So f* approaches W. However, a full 1.0 is often capped for safety.
            return max(0.0, win_probability) # Or a capped value like 0.5 for safety
        else: # No trades or no wins
            return 0.0
    
    # Ensure avg_gain is not zero if there are wins, to avoid division by zero in win_loss_ratio calculation later
    if avg_gain == 0 and num_wins > 0:
        print("Error: Average gain from winning trades is zero despite having wins. Cannot calculate R.")
        return 0.0


    win_loss_ratio = avg_gain / avg_loss

    # Calculate Kelly Fraction (f*)
    # f* = W - (1 - W) / R
    kelly_fraction = win_probability - (1 - win_probability) / win_loss_ratio

    return max(0.0, kelly_fraction)

# --- SMA Crossover Strategy Backtesting ---

def backtest_sma_crossover(
    symbol: str,
    start_date: str,
    end_date: str,
    short_window: int = 50,
    long_window: int = 200
) -> dict:

    print(f"\n--- Backtesting SMA Crossover for {symbol} ({short_window}/{long_window}) ---")
    print(f"Data Period: {start_date} to {end_date}")

    # 1. Download Historical Data
    try:
        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
        if data.empty:
            print(f"No data downloaded for {symbol}. Check symbol and date range.")
            return {
                'num_wins': 0, 'total_trades': 0,
                'total_gain_from_wins': 0.0, 'total_loss_from_losses': 0.0
            }
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        return {
            'num_wins': 0, 'total_trades': 0,
            'total_gain_from_wins': 0.0, 'total_loss_from_losses': 0.0
        }

    # 2. Calculate SMAs
    data[f'SMA_{short_window}'] = data['Close'].rolling(window=short_window).mean()
    data[f'SMA_{long_window}'] = data['Close'].rolling(window=long_window).mean()

    # Drop NaN values created by rolling window calculations
    data = data.dropna()

    if data.empty:
        print("Not enough data to calculate SMAs. Adjust date range or window sizes.")
        return {
            'num_wins': 0, 'total_trades': 0,
            'total_gain_from_wins': 0.0, 'total_loss_from_losses': 0.0
        }

    # 3. Generate Signals - Using .loc for safe assignment and avoiding SettingWithCopyWarning
    # Initialize 'Position' column with 0 (out of market)
    data['Position'] = 0

    # Set position to 1 (long) where short SMA > long SMA
    data.loc[data[f'SMA_{short_window}'] > data[f'SMA_{long_window}'], 'Position'] = 1

    # Set position to 0 (out) where short SMA < long SMA
    data.loc[data[f'SMA_{short_window}'] < data[f'SMA_{long_window}'], 'Position'] = 0

    # Calculate trade signals (1 for buy, -1 for sell, 0 otherwise)
    # A 'Trade_Signal' of 1 means a transition from 0 to 1 (buy)
    # A 'Trade_Signal' of -1 means a transition from 1 to 0 (sell)
    data['Trade_Signal'] = data['Position'].diff()

    # The first value of Trade_Signal will be NaN. Fill with 0 and ensure it's an integer type.
    # This explicit conversion should prevent any ambiguity issues.
    data['Trade_Signal'] = data['Trade_Signal'].fillna(0).astype(int)

    # --- Simulate Trades and Collect Statistics ---
    trades = []
    in_position = False
    entry_price = 0.0
    entry_date = None

    for i, row in data.iterrows():
        # IMPORTANT FIX: Explicitly get the scalar value from 'Trade_Signal'
        # This is to directly address the "truth value of a Series is ambiguous" error
        # by ensuring a scalar is used in the comparison.
        trade_signal_value = row['Trade_Signal'].item() # .item() extracts the scalar from a 0-d array or Series

        # Buy Signal: Trade_Signal is 1 and not currently in a position
        if trade_signal_value == 1 and not in_position:
            in_position = True
            entry_price = row['Close'].item() # Ensure entry_price is a scalar float
            entry_date = i
            # print(f"Buy Signal on {i.strftime('%Y-%m-%d')} at {entry_price:.2f}")

        # Sell Signal: Trade_Signal is -1 and currently in a position
        elif trade_signal_value == -1 and in_position:
            in_position = False
            exit_price = row['Close'].item() # Ensure exit_price is a scalar float
            profit_loss = exit_price - entry_price
            trades.append({
                'entry_date': entry_date,
                'exit_date': i,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_loss': profit_loss
            })
            # print(f"Sell Signal on {i.strftime('%Y-%m-%d')} at {exit_price:.2f}. P/L: {profit_loss:.2f}")

    # If still in a position at the end of the data, close it out
    if in_position:
        last_row = data.iloc[-1]
        exit_price = last_row['Close'].item() # Ensure exit_price is a scalar float here too
        profit_loss = exit_price - entry_price
        trades.append({
            'entry_date': entry_date,
            'exit_date': last_row.name,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_loss': profit_loss
        })
        # THIS IS THE LINE THAT WAS CAUSING THE ERROR
        print(f"Forced exit at end of data on {last_row.name.strftime('%Y-%m-%d')} at {exit_price:.2f}. P/L: {profit_loss:.2f}")


    # Aggregate results for Kelly Criterion
    num_wins = 0
    total_gain_from_wins = 0.0
    total_loss_from_losses = 0.0
    total_trades = len(trades)

    if total_trades == 0:
        print("No trades were executed based on this strategy within the given period.")
        return {
            'num_wins': 0, 'total_trades': 0,
            'total_gain_from_wins': 0.0, 'total_loss_from_losses': 0.0
        }

    for trade in trades:
        if trade['profit_loss'] > 0:
            num_wins += 1
            total_gain_from_wins += trade['profit_loss']
        else:
            total_loss_from_losses += abs(trade['profit_loss']) # Use absolute value for losses

    print("\n--- Backtest Summary ---")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {num_wins}")
    print(f"Losing Trades: {total_trades - num_wins}")
    print(f"Total Gain from Wins: ${total_gain_from_wins:.2f}")
    print(f"Total Loss from Losses: ${total_loss_from_losses:.2f}")
    print(f"Overall Net Profit/Loss: ${total_gain_from_wins - total_loss_from_losses:.2f}")

    return {
        'num_wins': num_wins,
        'total_trades': total_trades,
        'total_gain_from_wins': total_gain_from_wins,
        'total_loss_from_losses': total_loss_from_losses
    }

# --- Main Execution ---
if __name__ == "__main__":
    x = input()
    symbol_to_backtest = x
    start = "2019-01-01"
    end = "2025-07-20" # Today's date (or adjust as needed)

    kelly_inputs = backtest_sma_crossover(
        symbol=symbol_to_backtest,
        start_date=start,
        end_date=end,
        short_window=50,
        long_window=200
    )

    # Calculate Kelly Fraction using the results from the backtest
    kelly_fraction_result = calculate_kelly_fraction(
        num_wins=kelly_inputs['num_wins'],
        total_trades=kelly_inputs['total_trades'],
        total_gain_from_wins=kelly_inputs['total_gain_from_wins'],
        total_loss_from_losses=kelly_inputs['total_loss_from_losses']
    )

    print(f"\n--- Kelly Criterion Result for {symbol_to_backtest} SMA Strategy ---")
    print(f"Calculated Kelly Fraction: {kelly_fraction_result:.4f} (or {kelly_fraction_result * 100:.2f}%)")

    # Interpretation:
    if kelly_fraction_result > 0:
        print(f"This strategy appears to have a positive edge. The Kelly Criterion suggests risking up to {kelly_fraction_result * 100:.2f}% of your capital per trade based on historical performance.")
    else:
        print("This strategy does not appear to have a positive edge based on historical data.")
        print("The Kelly Criterion suggests risking 0% of your capital.")

    print("\n" + "="*70 + "\n")

    # Example with a different stock or period (e.g., MSFT)
    symbol_to_backtest_2 = "AMD"
    start_2 = "2018-01-01"
    end_2 = "2025-07-20" # A different period

    kelly_inputs_2 = backtest_sma_crossover(
        symbol=symbol_to_backtest_2,
        start_date=start_2,
        end_date=end_2,
        short_window=50,
        long_window=200
    )

    kelly_fraction_result_2 = calculate_kelly_fraction(
        num_wins=kelly_inputs_2['num_wins'],
        total_trades=kelly_inputs_2['total_trades'],
        total_gain_from_wins=kelly_inputs_2['total_gain_from_wins'],
        total_loss_from_losses=kelly_inputs_2['total_loss_from_losses']
    )

    print(f"\n--- Kelly Criterion Result for {symbol_to_backtest_2} SMA Strategy ---")
    print(f"Calculated Kelly Fraction: {kelly_fraction_result_2:.4f} (or {kelly_fraction_result_2 * 100:.2f}%)")

    if kelly_fraction_result_2 > 0:
        print(f"This strategy appears to have a positive edge. The Kelly Criterion suggests risking up to {kelly_fraction_result_2 * 100:.2f}% of your capital per trade based on historical performance.")
    else:
        print("This strategy does not appear to have a positive edge based on historical data.")
