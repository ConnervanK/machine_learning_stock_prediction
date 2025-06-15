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
                if (epoch+1) % 10 == 0 and idx_pos % 5 == 0:
                    print(f"Window {idx_pos}/{len(indices)}, Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")

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


def grid_search_hyperparameters(
    temporal_data,
    target_variable='Open',
    test_size=0.2,
    cv_folds=3,
    verbose=True,
    save_results=True
):
    """
    Perform grid search to find optimal hyperparameters for the financial prediction model.
    
    Args:
        temporal_data: Dictionary of loaded dataframes from load_and_plot_temporal_data
        target_variable: Variable to predict (e.g., 'interest', 'Open', 'Close')
        test_size: Proportion of data to use for final test evaluation
        cv_folds: Number of cross-validation folds for time series
        verbose: Whether to print detailed progress information
        save_results: Whether to save results to CSV
    
    Returns:
        best_params: Dictionary of best hyperparameters
        results_df: DataFrame with all grid search results
        best_model: The trained model with best parameters
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import time
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from datetime import datetime
    import torch
    import os
    
    print(f"Starting grid search for {target_variable} prediction...")
    start_time = time.time()
    
    # Create a directory for saving models and results if it doesn't exist
    save_dir = f"grid_search_results_{target_variable}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Define parameter grid
    param_grid = {
        'sequence_length': [10, 15, 20],
        'hidden_dim': [64, 128, 256],
        'num_layers': [2, 3, 4],
        'epochs': [20, 40],
        'batch_size': [16, 32],
        'use_features': [True, False]  # Whether to use multivariate features
    }
    
    # Calculate total number of parameter combinations
    total_combinations = 1
    for param, values in param_grid.items():
        total_combinations *= len(values)
    
    print(f"Grid search will test {total_combinations} parameter combinations")
    print(f"Estimated time: {total_combinations * 2:.1f} to {total_combinations * 5:.1f} minutes")
    
    # Prepare market data first to avoid redundant processing
    dates, data, target_idx, variable_names = prepare_market_data(
        temporal_data, target_variable=target_variable
    )
    
    # Create test/train split indices
    data_length = len(data)
    train_size = int(data_length * (1 - test_size))
    
    # For time series data, we need chronological splits
    cv_splits = []
    fold_size = int(train_size / (cv_folds + 1))
    
    for i in range(cv_folds):
        # For each fold, we use all data up to validation start for training
        # and a window after that for validation
        val_start = train_size - (i+1) * fold_size
        val_end = val_start + fold_size
        
        # Make sure we have enough training data
        train_cutoff = max(20, val_start - 3 * fold_size)
        
        cv_splits.append({
            'train_start': 0,
            'train_end': val_start,
            'val_start': val_start,
            'val_end': val_end
        })
    
    # Store results
    results = []
    best_score = float('inf')
    best_params = None
    best_model = None
    best_predictions = None
    best_actuals = None
    best_feature_cols = None
    best_scalers = None
    
    # Generate all parameter combinations
    from itertools import product
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # Function to evaluate a single parameter combination
    def evaluate_params(params_dict, fold):
        """Evaluate a single parameter combination on one fold"""
        # Extract train/val indices
        train_start = fold['train_start']
        train_end = fold['train_end']
        val_start = fold['val_start']
        val_end = fold['val_end']
        
        # Basic validation: ensure we have enough data for sequence
        if train_end - train_start < params_dict['sequence_length'] * 2:
            if verbose:
                print("Not enough training data for this sequence length")
            return float('inf'), float('inf'), None, None, None, None, None
        
        try:
            # Create a simplified single-fold version of the prediction
            from sklearn.preprocessing import MinMaxScaler
            
            # Create dataframe from data
            df = pd.DataFrame(data, columns=variable_names)
            df['date'] = dates
            df.set_index('date', inplace=True)
            
            # Add financial features
            df_with_features = add_financial_features(df, variable_names[target_idx])
            
            # Train on train_start:train_end, validate on val_start:val_end
            train_data = data[train_start:train_end]
            val_data = data[val_start:val_end]
            
            if params_dict['use_features']:
                # For multivariate: use all numeric features
                feature_cols = [col for col in df_with_features.select_dtypes(include=[np.number]).columns 
                              if col != variable_names[target_idx]]
                
                # Use only available rows after feature engineering
                train_features = df_with_features.iloc[train_start:train_end]
                val_features = df_with_features.iloc[val_start:val_end]
                
                # Scale features and target
                feature_scaler = MinMaxScaler()
                target_scaler = MinMaxScaler()
                
                # Prepare training data
                train_feature_data = train_features[feature_cols].values
                train_target_data = train_features[variable_names[target_idx]].values.reshape(-1, 1)
                
                # Scale training data
                scaled_train_features = feature_scaler.fit_transform(train_feature_data)
                scaled_train_target = target_scaler.fit_transform(train_target_data)
                
                # Prepare validation features
                val_feature_data = val_features[feature_cols].values
                val_target_data = val_features[variable_names[target_idx]].values.reshape(-1, 1)
                
                # Scale validation features using training scaler
                scaled_val_features = feature_scaler.transform(val_feature_data)
                scaled_val_target = target_scaler.transform(val_target_data)
                
                # Create sequences from the scaled training data
                X_train, y_train = [], []
                seq_len = params_dict['sequence_length']
                
                for i in range(len(scaled_train_features) - seq_len):
                    X_train.append(scaled_train_features[i:i+seq_len])
                    y_train.append(scaled_train_target[i+seq_len])
                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                
                # Train a simple LSTM model
                import torch
                import torch.nn as nn
                import torch.optim as optim
                
                # Define model
                class SimpleLSTM(nn.Module):
                    def __init__(self, input_dim, hidden_dim, num_layers=1):
                        super(SimpleLSTM, self).__init__()
                        self.hidden_dim = hidden_dim
                        self.num_layers = num_layers
                        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                        self.fc = nn.Linear(hidden_dim, 1)
                    
                    def forward(self, x):
                        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
                        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
                        out, _ = self.lstm(x, (h0, c0))
                        out = self.fc(out[:, -1, :])
                        return out
                
                # Create and train model
                input_dim = len(feature_cols)
                model = SimpleLSTM(
                    input_dim=input_dim, 
                    hidden_dim=params_dict['hidden_dim'], 
                    num_layers=params_dict['num_layers']
                )
                
                # Convert to tensors
                X_train_tensor = torch.FloatTensor(X_train)
                y_train_tensor = torch.FloatTensor(y_train)
                
                # Create data loader
                train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=params_dict['batch_size'], 
                    shuffle=True
                )
                
                # Define loss and optimizer
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                # Train the model
                for epoch in range(params_dict['epochs']):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                
                # Evaluate on validation set
                model.eval()
                val_predictions = []
                
                # We can only predict after having sequence_length previous observations
                for i in range(len(scaled_val_features) - seq_len):
                    # Get sequence
                    sequence = scaled_val_features[i:i+seq_len]
                    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
                    
                    # Make prediction
                    with torch.no_grad():
                        pred = model(sequence_tensor).numpy()[0, 0]
                        val_predictions.append(pred)
                
                # Convert predictions back to original scale
                val_predictions = np.array(val_predictions).reshape(-1, 1)
                val_predictions = target_scaler.inverse_transform(val_predictions).flatten()
                
                # Get actual values (offset by sequence length)
                val_actuals = target_scaler.inverse_transform(val_target_data[seq_len:]).flatten()
                
                # Calculate metrics
                mse = mean_squared_error(val_actuals, val_predictions)
                mae = mean_absolute_error(val_actuals, val_predictions)
                rmse = np.sqrt(mse)
                
                # Store scalers
                scalers = {
                    'feature_scaler': feature_scaler,
                    'target_scaler': target_scaler
                }
                
                return rmse, mae, model, val_predictions, val_actuals, feature_cols, scalers
                
            else:
                # Univariate prediction
                target_data = df[variable_names[target_idx]].values.reshape(-1, 1)
                target_scaler = MinMaxScaler()
                scaled_target = target_scaler.fit_transform(target_data)
                
                # Extract train and validation portions
                train_scaled = scaled_target[train_start:train_end]
                val_scaled = scaled_target[val_start:val_end]
                
                # Create sequences
                X_train, y_train = [], []
                seq_len = params_dict['sequence_length']
                
                for i in range(len(train_scaled) - seq_len):
                    X_train.append(train_scaled[i:i+seq_len])
                    y_train.append(train_scaled[i+seq_len])
                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                
                # Train a simple LSTM model for univariate prediction
                import torch
                import torch.nn as nn
                import torch.optim as optim
                
                # Define model
                class UnivariateRNN(nn.Module):
                    def __init__(self, hidden_dim, num_layers=1):
                        super(UnivariateRNN, self).__init__()
                        self.hidden_dim = hidden_dim
                        self.num_layers = num_layers
                        self.lstm = nn.LSTM(1, hidden_dim, num_layers, batch_first=True)
                        self.fc = nn.Linear(hidden_dim, 1)
                    
                    def forward(self, x):
                        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
                        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
                        out, _ = self.lstm(x, (h0, c0))
                        out = self.fc(out[:, -1, :])
                        return out
                
                # Create and train model
                model = UnivariateRNN(
                    hidden_dim=params_dict['hidden_dim'], 
                    num_layers=params_dict['num_layers']
                )
                
                # Reshape for univariate input
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                
                # Convert to tensors
                X_train_tensor = torch.FloatTensor(X_train)
                y_train_tensor = torch.FloatTensor(y_train)
                
                # Create data loader
                train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=params_dict['batch_size'], 
                    shuffle=True
                )
                
                # Define loss and optimizer
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                # Train the model
                for epoch in range(params_dict['epochs']):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                
                # Evaluate on validation set
                model.eval()
                val_predictions = []
                
                # We can only predict after having sequence_length previous observations
                for i in range(len(val_scaled) - seq_len):
                    # Get sequence
                    sequence = val_scaled[i:i+seq_len]
                    sequence_tensor = torch.FloatTensor(sequence).reshape(1, seq_len, 1)
                    
                    # Make prediction
                    with torch.no_grad():
                        pred = model(sequence_tensor).numpy()[0, 0]
                        val_predictions.append(pred)
                
                # Convert predictions back to original scale
                val_predictions = np.array(val_predictions).reshape(-1, 1)
                val_predictions = target_scaler.inverse_transform(val_predictions).flatten()
                
                # Get actual values (offset by sequence length)
                val_actuals = target_scaler.inverse_transform(val_scaled[seq_len:]).flatten()
                
                # Calculate metrics
                mse = mean_squared_error(val_actuals, val_predictions)
                mae = mean_absolute_error(val_actuals, val_predictions)
                rmse = np.sqrt(mse)
                
                # Store scalers (no feature scaler for univariate)
                scalers = {
                    'feature_scaler': None,
                    'target_scaler': target_scaler
                }
                
                # For univariate, feature_cols is None
                return rmse, mae, model, val_predictions, val_actuals, None, scalers
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error evaluating parameters: {str(e)}")
            return float('inf'), float('inf'), None, None, None, None, None
    
    # Iterate through all parameter combinations
    for i, params_values in enumerate(product(*param_values)):
        params_dict = dict(zip(param_keys, params_values))
        
        if verbose:
            print(f"\nEvaluating combination {i+1}/{total_combinations}:")
            for k, v in params_dict.items():
                print(f"  {k}: {v}")
        
        # Perform cross-validation
        cv_scores_rmse = []
        cv_scores_mae = []
        cv_models = []
        cv_predictions = []
        cv_actuals = []
        cv_feature_cols = []
        cv_scalers = []
        
        for fold_idx, fold in enumerate(cv_splits):
            if verbose:
                print(f"  Fold {fold_idx+1}/{len(cv_splits)}...")
            
            fold_rmse, fold_mae, fold_model, fold_preds, fold_actuals, fold_feature_cols, fold_scalers = evaluate_params(params_dict, fold)
            
            if fold_rmse < float('inf'):
                cv_scores_rmse.append(fold_rmse)
                cv_scores_mae.append(fold_mae)
                cv_models.append(fold_model)
                cv_predictions.append(fold_preds)
                cv_actuals.append(fold_actuals)
                cv_feature_cols.append(fold_feature_cols)
                cv_scalers.append(fold_scalers)
        
        # Calculate mean scores across folds
        if len(cv_scores_rmse) > 0:
            mean_rmse = np.mean(cv_scores_rmse)
            mean_mae = np.mean(cv_scores_mae)
            std_rmse = np.std(cv_scores_rmse)
            
            if verbose:
                print(f"  Mean RMSE: {mean_rmse:.4f} (±{std_rmse:.4f})")
                print(f"  Mean MAE: {mean_mae:.4f}")
        else:
            mean_rmse = float('inf')
            mean_mae = float('inf')
            std_rmse = float('inf')
            
            if verbose:
                print("  Failed to complete cross-validation")
        
        # Store result
        result = {
            **params_dict,
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'mean_mae': mean_mae,
            'num_folds_completed': len(cv_scores_rmse)
        }
        results.append(result)
        
        # Update best parameters if better
        if mean_rmse < best_score and len(cv_scores_rmse) == len(cv_splits):
            best_score = mean_rmse
            best_params = params_dict.copy()
            
            # Find the best fold (lowest RMSE)
            best_fold_idx = np.argmin(cv_scores_rmse)
            best_model = cv_models[best_fold_idx]
            best_predictions = cv_predictions[best_fold_idx]
            best_actuals = cv_actuals[best_fold_idx]
            best_feature_cols = cv_feature_cols[best_fold_idx]
            best_scalers = cv_scalers[best_fold_idx]
            
            if verbose:
                print(f"  New best! RMSE: {mean_rmse:.4f}")
                
                # Save the best model after each improvement
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(save_dir, f"best_model_{target_variable}_{timestamp}.pth")
                torch.save(best_model.state_dict(), model_path)
                print(f"  Saved new best model to {model_path}")
                
                # Also save the best predictions and actuals
                pred_df = pd.DataFrame({
                    'Predicted': best_predictions,
                    'Actual': best_actuals
                })
                pred_path = os.path.join(save_dir, f"best_predictions_{target_variable}_{timestamp}.csv")
                pred_df.to_csv(pred_path)
                print(f"  Saved predictions to {pred_path}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by mean_rmse
    results_df = results_df.sort_values('mean_rmse')
    
    # Print top results
    print("\nTop 5 Parameter Combinations:")
    print(results_df.head(5).to_string())
    
    # Print best parameters
    print("\nBest Parameters Found:")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    
    # Print best score
    print(f"Best RMSE: {best_score:.4f}")
    
    # Save final results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"grid_search_results_{target_variable}_{timestamp}.csv")
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        
        # Save best model metadata
        metadata = {
            'target_variable': target_variable,
            'best_params': best_params,
            'best_rmse': best_score,
            'feature_columns': best_feature_cols if best_params['use_features'] else None,
            'timestamp': timestamp
        }
        meta_path = os.path.join(save_dir, f"best_model_metadata_{target_variable}_{timestamp}.json")
        
        import json
        with open(meta_path, 'w') as f:
            # Convert any numpy types to native Python types
            meta_dict = {}
            for k, v in metadata.items():
                if isinstance(v, dict):
                    meta_dict[k] = {k2: v2 if not isinstance(v2, np.generic) else v2.item() 
                                   for k2, v2 in v.items()}
                else:
                    meta_dict[k] = v if not isinstance(v, np.generic) else v.item()
            
            json.dump(meta_dict, f, indent=4)
        print(f"Model metadata saved to {meta_path}")
        
        # Save scalers using pickle
        import pickle
        if best_scalers:
            scaler_path = os.path.join(save_dir, f"best_scalers_{target_variable}_{timestamp}.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(best_scalers, f)
            print(f"Scalers saved to {scaler_path}")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Grid search completed in {elapsed_time/60:.2f} minutes")
    
    # Create a function to load the best model for future use
    def load_best_model():
        """Helper function to reload the best model with correct architecture"""
        if best_params['use_features']:
            # Multivariate model
            class SimpleLSTM(nn.Module):
                def __init__(self, input_dim, hidden_dim, num_layers=1):
                    super(SimpleLSTM, self).__init__()
                    self.hidden_dim = hidden_dim
                    self.num_layers = num_layers
                    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_dim, 1)
                
                def forward(self, x):
                    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
                    out, _ = self.lstm(x, (h0, c0))
                    out = self.fc(out[:, -1, :])
                    return out
            
            input_dim = len(best_feature_cols) if best_feature_cols else 0
            loaded_model = SimpleLSTM(
                input_dim=input_dim,
                hidden_dim=best_params['hidden_dim'],
                num_layers=best_params['num_layers']
            )
        else:
            # Univariate model
            class UnivariateRNN(nn.Module):
                def __init__(self, hidden_dim, num_layers=1):
                    super(UnivariateRNN, self).__init__()
                    self.hidden_dim = hidden_dim
                    self.num_layers = num_layers
                    self.lstm = nn.LSTM(1, hidden_dim, num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_dim, 1)
                
                def forward(self, x):
                    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
                    out, _ = self.lstm(x, (h0, c0))
                    out = self.fc(out[:, -1, :])
                    return out
                
            loaded_model = UnivariateRNN(
                hidden_dim=best_params['hidden_dim'],
                num_layers=best_params['num_layers']
            )
            
        # Copy the state_dict from the best model
        loaded_model.load_state_dict(best_model.state_dict())
        return loaded_model
    
    # Return the best model
    return best_params, results_df, best_model, best_scalers, best_feature_cols

# # Load temporal data if not already loaded
# if 'temporal_data' not in globals():
#     temporal_data = load_and_plot_temporal_data('data')

# # Run grid search for interest rate prediction
# best_params, results_df, best_model, best_scalers, best_feature_cols = grid_search_hyperparameters(
#     temporal_data,
#     target_variable='Close',  # Change to the target variable you want to predict
#     test_size=0.2,
#     cv_folds=3,  # Use 3 time-series validation folds
#     verbose=True,
#     save_results=True
# )

# # Visualize the grid search results
# plt.figure(figsize=(15, 10))

# # Plot the influence of each parameter on RMSE
# param_names = ['sequence_length', 'hidden_dim', 'num_layers', 'epochs', 'batch_size', 'use_features']
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# axes = axes.flatten()

# for i, param in enumerate(param_names):
#     sns.boxplot(x=param, y='mean_rmse', data=results_df, ax=axes[i])
#     axes[i].set_title(f'Effect of {param} on RMSE')
#     axes[i].set_ylabel('RMSE')
#     if param == 'use_features':
#         axes[i].set_xticklabels(['Univariate', 'Multivariate'])

# plt.tight_layout()
# plt.savefig(f'grid_search_parameter_effects_{target_variable}.png')
# plt.show()

# # Run prediction with best parameters and save the full model
# print("\nRunning final prediction with best parameters:")
# dates, predictions, actuals, std_devs = parallelized_rolling_window_prediction_for_financial_data(
#     temporal_data,
#     target_variable='interest',  # Use the same target variable as in grid search
#     sequence_length=best_params['sequence_length'],
#     epochs=best_params['epochs'],
#     hidden_dim=best_params['hidden_dim'],
#     num_layers=best_params['num_layers'],
#     batch_size=best_params['batch_size'],
#     use_features=best_params['use_features'],
#     mc_samples=30  # Keep this consistent
# )

# # Save the full final model with best parameters
# from datetime import datetime
# import torch
# import os

# save_dir = f"final_models"
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# model_filename = f"final_model_{target_variable}_{timestamp}.pth"
# model_path = os.path.join(save_dir, model_filename)

# # Save the model state dict
# torch.save({
#     'model_state_dict': best_model.state_dict(),
#     'parameters': best_params,
#     'feature_columns': best_feature_cols,
#     'target_variable': target_variable
# }, model_path)

# print(f"Final model saved to {model_path}")

# # Save the final predictions
# results_df = pd.DataFrame({
#     'Date': dates,
#     'Actual': actuals,
#     'Predicted': predictions,
#     'Error': np.array(actuals) - np.array(predictions),
#     'Uncertainty': std_devs * 2  # 95% confidence interval
# })

# results_filename = f"final_predictions_{target_variable}_{timestamp}.csv"
# results_path = os.path.join(save_dir, results_filename)
# results_df.to_csv(results_path, index=False)

# print(f"Final predictions saved to {results_path}")

# # Save scalers for future inference
# if best_scalers:
#     import pickle
#     scaler_filename = f"final_scalers_{target_variable}_{timestamp}.pkl"
#     scaler_path = os.path.join(save_dir, scaler_filename)
#     with open(scaler_path, 'wb') as f:
#         pickle.dump(best_scalers, f)
#     print(f"Scalers saved to {scaler_path}")