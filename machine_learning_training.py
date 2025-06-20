import machine_learning_data as mld

def run_data_test():
    """
    Test function to ensure the module is working correctly
    """
    print("Machine learning training module is working correctly.")
    mld.test()

def parallelized_rolling_window_prediction_for_financial_data(
    temporal_data, 
    target_variable='Open',
    initial_train_size=0.7,
    sequence_length=20,
    epochs=30,
    hidden_dim=64,
    num_layers=4,
    batch_size=32,
    num_workers=4,
    window_step=1,
    mc_samples=50,
    use_features=True
):
    """
    Implements parallelized rolling window forecasting with Monte Carlo dropout uncertainty
    estimation for financial market data.
    
    Args:
        temporal_data: Dictionary of loaded dataframes from load_and_plot_temporal_data
        target_variable: Variable to predict (default: 'Open')
        initial_train_size: Proportion of data used for initial training
        sequence_length: Number of time steps to use as input
        epochs: Number of training epochs per window
        hidden_dim: Hidden dimension for LSTM layers
        num_layers: Number of LSTM layers
        batch_size: Training batch size
        num_workers: Number of parallel workers
        window_step: Step size between prediction windows
        mc_samples: Number of Monte Carlo samples for uncertainty estimation
        use_features: Whether to use multivariate features
        
    Returns:
        dates, predictions, actuals, std_devs: Prediction results and uncertainty
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    import time
    from tqdm import tqdm
    import concurrent.futures
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import machine_learning_data as mld
    import pandas as pd

    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare market data using the existing function
    print(f"Preparing market data for predicting '{target_variable}'...")
    dates, data, target_idx, variable_names = mld.prepare_market_data(
        temporal_data, target_variable=target_variable
    )
    
    # Create DataFrame with the aligned data
    df = pd.DataFrame(data, columns=variable_names)
    df['date'] = dates
    df.set_index('date', inplace=True)
    
    target_column = variable_names[target_idx]
    print(f"Target column: {target_column}")
    
    # Add financial engineering features
    print("Adding financial features...")
    df = add_financial_features(df, target_column)
    
    if use_features:
        # Use all numeric features except the target for multivariate prediction
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                      if col != target_column]
        print(f"Using {len(feature_cols)} features for prediction")
        
        # Scale features and target separately
        target_data = df[target_column].values.reshape(-1, 1)
        feature_data = df[feature_cols].values
        
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        scaled_features = feature_scaler.fit_transform(feature_data)
        scaled_target = target_scaler.fit_transform(target_data)
        
        # Input dimension is number of features
        input_dim = len(feature_cols)
    else:
        # Univariate prediction
        print("Using univariate prediction (target only)")
        target_data = df[target_column].values.reshape(-1, 1)
        target_scaler = MinMaxScaler()
        scaled_target = target_scaler.fit_transform(target_data)
        
        # Input dimension is 1 (univariate)
        input_dim = 1

    # Calculate initial split index
    train_end_idx = int(len(df) * initial_train_size)
    print(f"Training on {train_end_idx} samples, predicting {len(df) - train_end_idx}")

    # Define LSTM model with dropout for Monte Carlo sampling
    class FinancialLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.3, output_dim=1):
            super(FinancialLSTM, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.dropout = dropout
            
            self.lstm = nn.LSTM(
                input_dim, 
                hidden_dim, 
                num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.drop_layer = nn.Dropout(p=dropout)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.drop_layer(out[:, -1, :])  # Enable dropout for inference uncertainty
            return self.fc(out)

    # Prepare the window indices
    indices = list(range(train_end_idx, len(df) - 1, window_step))
    results = [None] * len(indices)

    # Function to create sequences
    def create_financial_sequences(target_data, feature_data=None, seq_length=20):
        X, y = [], []
        
        for i in range(len(target_data) - seq_length):
            if feature_data is not None:
                # Multivariate
                X.append(feature_data[i:i+seq_length])
            else:
                # Univariate
                X.append(target_data[i:i+seq_length])
                
            y.append(target_data[i+seq_length])
            
        return np.array(X), np.array(y)

    # Process each window in parallel
    def process_window(idx_pos, i):
        try:
            if use_features:
                # Prepare multivariate data
                current_target = scaled_target[:i+1]
                current_features = scaled_features[:i+1]
                
                # Create sequences
                X, y = create_financial_sequences(
                    current_target, current_features, sequence_length
                )
                
                # Reshape for LSTM: [samples, time steps, features]
                X = X.reshape(X.shape[0], sequence_length, input_dim)
            else:
                # Prepare univariate data
                current_data = scaled_target[:i+1]
                
                # Create sequences
                X, y = [], []
                for j in range(len(current_data) - sequence_length):
                    X.append(current_data[j:j+sequence_length])
                    y.append(current_data[j+sequence_length])
                
                X = np.array(X)
                y = np.array(y)

            # Convert to tensors
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)

            # Create model and optimizer
            model = FinancialLSTM(
                input_dim=input_dim, 
                hidden_dim=hidden_dim, 
                num_layers=num_layers
            ).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Create data loader
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_tensor, y_tensor),
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True
            )

            # Training loop
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                # Report progress occasionally
                # if (epoch+1) % 10 == 0 and idx_pos % 5 == 0:
                #     print(f"Window {idx_pos}/{len(indices)}, Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")
            # Report final loss for this window# Report progress less frequently
                if (epoch+1) == epochs and idx_pos % 25 == 0:
                    print(f"Window {idx_pos}/{len(indices)}: Final loss {total_loss/len(train_loader):.4f}")

            # Prepare for prediction
            if use_features:
                # For multivariate, use the last sequence of features
                last_sequence = scaled_features[i+1-sequence_length:i+1]
                last_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                # For univariate, use the last sequence of the target
                last_sequence = scaled_target[i+1-sequence_length:i+1]
                last_sequence = torch.tensor(last_sequence, dtype=torch.float32).reshape(1, sequence_length, 1).to(device)

            # Monte Carlo Dropout Inference
            model.train()  # Keep dropout active for MC sampling
            preds = []
            
            with torch.no_grad():
                for _ in range(mc_samples):
                    pred = model(last_sequence).cpu().numpy()
                    preds.append(pred)

            # Calculate mean and std of predictions
            preds = np.array(preds).squeeze()
            mean_pred = preds.mean()
            std_pred = preds.std()

            # Convert back to original scale
            mean_pred_unscaled = target_scaler.inverse_transform([[mean_pred]])[0][0]
            actual_next_day = target_scaler.inverse_transform(scaled_target[i+1:i+2])[0][0]
            
            # Scale the standard deviation to match the original scale
            std_unscaled = std_pred * (target_scaler.data_max_[0] - target_scaler.data_min_[0])

            # Store results
            results[idx_pos] = (
                df.index[i+1],
                mean_pred_unscaled,
                actual_next_day,
                std_unscaled
            )
            return True
            
        except Exception as e:
            print(f"Error processing window {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    # Parallel processing of windows
    print(f"\nStarting parallel processing with {num_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_window, idx, i) for idx, i in enumerate(indices)]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    # Process results
    results = [r for r in results if r is not None]
    if not results:
        print("No predictions were made.")
        return [], [], [], []

    # Unpack results
    dates, predictions, actuals, std_devs = zip(*results)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100

    # Plot results
    plt.figure(figsize=(14, 7))
    predictions = np.array(predictions)
    std_devs = np.array(std_devs)

    plt.plot(dates, actuals, label='Actual', color='blue')
    plt.plot(dates, predictions, label='Predicted', color='red', linestyle='--')
    plt.fill_between(dates,
                    predictions - 2 * std_devs,
                    predictions + 2 * std_devs,
                    color='red', alpha=0.2,
                    label='±2σ Uncertainty')
    plt.title(f"{target_column} Prediction with MC Dropout Uncertainty\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    plt.xlabel('Date')
    plt.ylabel(f'{target_column} Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.savefig(f'parallel_mc_prediction_{target_column}.png')
    plt.show()
    
    # Create a scatter plot of actual vs predicted values
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.6)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'k--')
    plt.title(f'Actual vs Predicted {target_column} Values\nR²: {np.corrcoef(actuals, predictions)[0,1]**2:.4f}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'parallel_scatter_{target_column}.png')
    plt.show()

    # Print performance metrics
    elapsed = time.time() - start_time
    print(f"Parallel Processing Completed in {elapsed:.2f} seconds")
    print(f"Prediction Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Correlation: {np.corrcoef(actuals, predictions)[0,1]:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Date': dates,
        'Actual': actuals,
        'Predicted': predictions,
        'Error': np.array(actuals) - np.array(predictions),
        'Uncertainty': std_devs * 2  # 95% confidence interval
    })
    results_df.to_csv(f'parallel_mc_results_{target_column}.csv', index=False)
    print(f"Results saved to parallel_mc_results_{target_column}.csv")

    return dates, predictions, actuals, std_devs

# Helper function for adding financial features
def add_financial_features(df, target_column):
    """Add financial-specific features to the dataframe"""
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    df = df.copy()
    
    try:
        # Log returns
        df[f'{target_column}_return'] = df[target_column].pct_change()
        
        # Moving averages
        for window in [5, 10, 20]:
            df[f'{target_column}_ma_{window}'] = df[target_column].rolling(window=window).mean()
            
            # Add distance from moving average
            df[f'{target_column}_dist_ma_{window}'] = df[target_column] - df[f'{target_column}_ma_{window}']
        
        # Volatility (standard deviation)
        for window in [10, 20]:
            df[f'{target_column}_std_{window}'] = df[target_column].rolling(window=window).std()
        
        # Previous values (lags)
        for lag in range(1, 6):
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        
        # Time-based features (if index is datetime)
        if isinstance(df.index[0], (pd.Timestamp, datetime)):
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            
            # Cyclical encoding of time features
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
        
    except Exception as e:
        print(f"Warning: Could not create some financial features: {e}")
    
    # Drop rows with NaN values from lag features
    df = df.dropna()
    
    return df


def parallelized_rolling_window_prediction_for_financial_data2(
    tensor_data, 
    target_variable='Open',
    initial_train_size=0.7,
    sequence_length=20,
    epochs=30,
    hidden_dim=64,
    num_layers=4,
    batch_size=32,
    num_workers=4,
    window_step=1,
    mc_samples=50,
    use_features=True,
    merged_df=None
):
    """
    Implements parallelized rolling window forecasting with Monte Carlo dropout uncertainty
    estimation for financial market data.
    
    Args:
        tensor_data: Tensor data from create_tensor_from_csvs or dictionary of dataframes
        target_variable: Variable to predict (default: 'Open')
        initial_train_size: Proportion of data used for initial training
        sequence_length: Number of time steps to use as input
        epochs: Number of training epochs per window
        hidden_dim: Hidden dimension for LSTM layers
        num_layers: Number of LSTM layers
        batch_size: Training batch size
        num_workers: Number of parallel workers
        window_step: Step size between prediction windows
        mc_samples: Number of Monte Carlo samples for uncertainty estimation
        use_features: Whether to use multivariate features
        merged_df: Optional dataframe from create_tensor_from_csvs (if not provided, will attempt to extract from tensor_data)
        
    Returns:
        dates, predictions, actuals, std_devs: Prediction results and uncertainty
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    import time
    from tqdm import tqdm
    import concurrent.futures
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import machine_learning_data as mld
    import pandas as pd

    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Handle tensor data from create_tensor_from_csvs
    if isinstance(tensor_data, torch.Tensor):
        print("Processing tensor data from create_tensor_from_csvs...")
        
        # If merged_df is not provided, raise error as we need it to identify columns
        if merged_df is None:
            try:
                # Try to get the second return value from global namespace
                import inspect
                frame = inspect.currentframe()
                try:
                    # Look for merged_df or loaded_data in the caller's namespace
                    if 'loaded_data' in frame.f_back.f_locals:
                        merged_df = frame.f_back.f_locals['loaded_data']
                        print("Automatically detected loaded_data from caller's namespace")
                    else:
                        raise ValueError("merged_df not provided and could not be auto-detected. "
                                        "Please provide the merged DataFrame from create_tensor_from_csvs")
                finally:
                    del frame  # Avoid reference cycles
            except:
                raise ValueError("merged_df not provided and could not be auto-detected. "
                               "Please provide the merged DataFrame from create_tensor_from_csvs")
        
        # Check if target variable exists in the DataFrame
        if target_variable not in merged_df.columns:
            raise ValueError(f"Target variable '{target_variable}' not found in the data. "
                           f"Available columns: {merged_df.columns.tolist()}")
        
        # Get dates from the merged_df
        dates = merged_df['date'] if 'date' in merged_df.columns else merged_df.index
        
        # Get column indices from the merged_df
        target_idx = merged_df.columns.get_loc(target_variable) if target_variable in merged_df.columns else -1
        variable_names = merged_df.columns.tolist()
        
        # Convert tensor to numpy array
        data = tensor_data.numpy()
        
        # Create DataFrame with the aligned data
        df = merged_df.copy()
        
        # Make sure date is the index
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
        
    else:
        # Original functionality for dictionary of dataframes
        print(f"Preparing market data for predicting '{target_variable}'...")
        dates, data, target_idx, variable_names = mld.prepare_market_data(
            tensor_data, target_variable=target_variable
        )
        
        # Create DataFrame with the aligned data
        df = pd.DataFrame(data, columns=variable_names)
        df['date'] = dates
        df.set_index('date', inplace=True)
    
    target_column = target_variable
    print(f"Target column: {target_column}")
    
    # Add financial engineering features
    print("Adding financial features...")
    df = add_financial_features(df, target_column)
    
    if use_features:
        # Use all numeric features except the target for multivariate prediction
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                      if col != target_column]
        print(f"Using {len(feature_cols)} features for prediction")
        
        # Scale features and target separately
        target_data = df[target_column].values.reshape(-1, 1)
        feature_data = df[feature_cols].values
        
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        scaled_features = feature_scaler.fit_transform(feature_data)
        scaled_target = target_scaler.fit_transform(target_data)
        
        # Input dimension is number of features
        input_dim = len(feature_cols)
    else:
        # Univariate prediction
        print("Using univariate prediction (target only)")
        target_data = df[target_column].values.reshape(-1, 1)
        target_scaler = MinMaxScaler()
        scaled_target = target_scaler.fit_transform(target_data)
        
        # Input dimension is 1 (univariate)
        input_dim = 1

    # Calculate initial split index
    train_end_idx = int(len(df) * initial_train_size)
    print(f"Training on {train_end_idx} samples, predicting {len(df) - train_end_idx}")

    # Define LSTM model with dropout for Monte Carlo sampling
    class FinancialLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.3, output_dim=1):
            super(FinancialLSTM, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.dropout = dropout
            
            self.lstm = nn.LSTM(
                input_dim, 
                hidden_dim, 
                num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.drop_layer = nn.Dropout(p=dropout)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.drop_layer(out[:, -1, :])  # Enable dropout for inference uncertainty
            return self.fc(out)

    # Prepare the window indices
    indices = list(range(train_end_idx, len(df) - 1, window_step))
    results = [None] * len(indices)

    # Function to create sequences
    def create_financial_sequences(target_data, feature_data=None, seq_length=20):
        X, y = [], []
        
        for i in range(len(target_data) - seq_length):
            if feature_data is not None:
                # Multivariate
                X.append(feature_data[i:i+seq_length])
            else:
                # Univariate
                X.append(target_data[i:i+seq_length])
                
            y.append(target_data[i+seq_length])
            
        return np.array(X), np.array(y)

    # Process each window in parallel
    def process_window(idx_pos, i):
        try:
            if use_features:
                # Prepare multivariate data
                current_target = scaled_target[:i+1]
                current_features = scaled_features[:i+1]
                
                # Create sequences
                X, y = create_financial_sequences(
                    current_target, current_features, sequence_length
                )
                
                # Reshape for LSTM: [samples, time steps, features]
                X = X.reshape(X.shape[0], sequence_length, input_dim)
            else:
                # Prepare univariate data
                current_data = scaled_target[:i+1]
                
                # Create sequences
                X, y = [], []
                for j in range(len(current_data) - sequence_length):
                    X.append(current_data[j:j+sequence_length])
                    y.append(current_data[j+sequence_length])
                
                X = np.array(X)
                y = np.array(y)

            # Convert to tensors
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)

            # Create model and optimizer
            model = FinancialLSTM(
                input_dim=input_dim, 
                hidden_dim=hidden_dim, 
                num_layers=num_layers
            ).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Create data loader
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_tensor, y_tensor),
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True
            )

            # Training loop
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                # Report progress less frequently
                if (epoch+1) == epochs and idx_pos % 25 == 0:
                    print(f"Window {idx_pos}/{len(indices)}: Final loss {total_loss/len(train_loader):.4f}")

            # Prepare for prediction
            if use_features:
                # For multivariate, use the last sequence of features
                last_sequence = scaled_features[i+1-sequence_length:i+1]
                last_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                # For univariate, use the last sequence of the target
                last_sequence = scaled_target[i+1-sequence_length:i+1]
                last_sequence = torch.tensor(last_sequence, dtype=torch.float32).reshape(1, sequence_length, 1).to(device)

            # Monte Carlo Dropout Inference
            model.train()  # Keep dropout active for MC sampling
            preds = []
            
            with torch.no_grad():
                for _ in range(mc_samples):
                    pred = model(last_sequence).cpu().numpy()
                    preds.append(pred)

            # Calculate mean and std of predictions
            preds = np.array(preds).squeeze()
            mean_pred = preds.mean()
            std_pred = preds.std()

            # Convert back to original scale
            mean_pred_unscaled = target_scaler.inverse_transform([[mean_pred]])[0][0]
            actual_next_day = target_scaler.inverse_transform(scaled_target[i+1:i+2])[0][0]
            
            # Scale the standard deviation to match the original scale
            std_unscaled = std_pred * (target_scaler.data_max_[0] - target_scaler.data_min_[0])

            # Store results
            results[idx_pos] = (
                df.index[i+1],
                mean_pred_unscaled,
                actual_next_day,
                std_unscaled
            )
            return True
            
        except Exception as e:
            print(f"Error processing window {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    # Parallel processing of windows
    print(f"\nStarting parallel processing with {num_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_window, idx, i) for idx, i in enumerate(indices)]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    # Process results
    results = [r for r in results if r is not None]
    if not results:
        print("No predictions were made.")
        return [], [], [], []

    # Unpack results
    dates, predictions, actuals, std_devs = zip(*results)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100

    # Plot results
    plt.figure(figsize=(14, 7))
    predictions = np.array(predictions)
    std_devs = np.array(std_devs)

    plt.plot(dates, actuals, label='Actual', color='blue')
    plt.plot(dates, predictions, label='Predicted', color='red', linestyle='--')
    plt.fill_between(dates,
                    predictions - 2 * std_devs,
                    predictions + 2 * std_devs,
                    color='red', alpha=0.2,
                    label='±2σ Uncertainty')
    plt.title(f"{target_column} Prediction with MC Dropout Uncertainty\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    plt.xlabel('Date')
    plt.ylabel(f'{target_column} Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.savefig(f'parallel_mc_prediction_{target_column}.png')
    plt.show()
    
    # Create a scatter plot of actual vs predicted values
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.6)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'k--')
    plt.title(f'Actual vs Predicted {target_column} Values\nR²: {np.corrcoef(actuals, predictions)[0,1]**2:.4f}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'parallel_scatter_{target_column}.png')
    plt.show()

    # Print performance metrics
    elapsed = time.time() - start_time
    print(f"Parallel Processing Completed in {elapsed:.2f} seconds")
    print(f"Prediction Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Correlation: {np.corrcoef(actuals, predictions)[0,1]:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Date': dates,
        'Actual': actuals,
        'Predicted': predictions,
        'Error': np.array(actuals) - np.array(predictions),
        'Uncertainty': std_devs * 2  # 95% confidence interval
    })
    results_df.to_csv(f'parallel_mc_results_{target_column}.csv', index=False)
    print(f"Results saved to parallel_mc_results_{target_column}.csv")

    return dates, predictions, actuals, std_devs