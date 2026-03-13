# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pandas as pd
import numpy as np
import yaml
import os
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

class CaTSGDatasetCreator:
    """
    Creates TSG datasets from raw data based on configuration files.
    Handles preprocessing, feature engineering, and data splitting for all TSG tasks.
    """
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()
        self.df = None
        self.scalers = {}
        self.encoders = {}
        self._setup_variable_classifications()
        
    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_variable_classifications(self):
        """
        Pre-setup variable classifications based on configuration.
        This optimizes repeated lookups during data processing.
        """
        preprocessing_config = self.config.get('preprocessing', {})
        split_method = preprocessing_config.get('split_method')
        
        if split_method is None:
            source_type = self.config.get('dataset', {}).get('name')
            if source_type == 'harmonic_vm':
                split_method = 'alpha_based'
            elif source_type == 'harmonic_vp':
                split_method = 'combination_based'
        
        self.c_var_names = self._get_condition_variables(split_method)

        feature_embeddings = self.config.get('feature_embeddings', {})
        self.c_cat_names = list(feature_embeddings.keys())
        self.c_binary_names = self._get_binary_variable_names()
        
        self.c_continuous_names = [
            var for var in self.c_var_names 
            if var not in self.c_cat_names and var not in self.c_binary_names
        ]
        
        # Create index mappings for quick lookup
        self.var_to_index = {var: idx for idx, var in enumerate(self.c_var_names)}
        self.cat_indices = {self.var_to_index[var] for var in self.c_cat_names if var in self.var_to_index}
        self.binary_indices = {self.var_to_index[var] for var in self.c_binary_names if var in self.var_to_index}
        self.continuous_indices = {self.var_to_index[var] for var in self.c_continuous_names if var in self.var_to_index}
        
        print(f"Variable Classifications Setup:")
        print(f"   Condition Variables: {self.c_var_names}")
        print(f"   Categorical ({len(self.c_cat_names)}): {self.c_cat_names}")
        print(f"   Binary ({len(self.c_binary_names)}): {self.c_binary_names}")
        print(f"   Continuous ({len(self.c_continuous_names)}): {self.c_continuous_names}")
    
    def _get_binary_variable_names(self):
        """
        Get list of binary variable names that need special handling.
        These are variables treated as binary (0/1) but not in feature_embeddings.
        """
        known_binary_vars = ['holiday']
        binary_vars = [var for var in known_binary_vars if var in self.c_var_names]
        return binary_vars
    
    def _setup_output_dirs(self):
        """Create output directory structure."""
        output_config = self.config.get('output', {})
        base_dir = output_config.get('base_dir', './dataset')
        dataset_name = self.config['dataset']['name'].lower().replace(' ', '_')
        
        output_root = f"{base_dir}/{dataset_name}"
        os.makedirs(output_root, exist_ok=True)
        
        return output_root
    
    def _validate_and_clean_numeric_column(self, df, column, min_val=None, max_val=None, replace_method='median'):
        """
        Helper method to validate and clean numeric columns across all datasets.
        """
        if column not in df.columns:
            return df
        
        # Identify invalid values
        invalid_mask = pd.Series(False, index=df.index)
        
        if min_val is not None:
            invalid_mask |= df[column] < min_val
        if max_val is not None:
            invalid_mask |= df[column] > max_val
        
        # Also check for NaN values
        invalid_mask |= df[column].isna()
        
        if invalid_mask.any():
            if replace_method == 'median':
                replacement_value = df.loc[~invalid_mask, column].median()
            elif replace_method == 'mean':
                replacement_value = df.loc[~invalid_mask, column].mean()
            else:  # constant
                replacement_value = 0
            
            df.loc[invalid_mask, column] = replacement_value
            print(f"Cleaned {invalid_mask.sum()} invalid values in {column}")
        
        return df
    
    def _load_multi_station_data(self):
        """
        Load multi-station data from directory with file pattern.
        """
        import glob
        
        source_config = self.config['dataset']['source']
        data_directory = source_config['directory']
        file_pattern = source_config['file_pattern']
        
        # Find all matching files
        pattern_path = os.path.join(data_directory, file_pattern)
        station_files = glob.glob(pattern_path)
        
        if not station_files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern_path}")
        
        print(f"Found {len(station_files)} station files")
        
        # Load all station data
        all_stations = []
        for file_path in station_files:
            filename = os.path.basename(file_path)
            station_name = filename.replace('PRSA_Data_', '').replace('_20130301-20170228.csv', '')
            
            print(f"  Loading station: {station_name}")
            
            df_station = pd.read_csv(file_path)
            df_station['station'] = station_name
            
            all_stations.append(df_station)
        
        df = pd.concat(all_stations, ignore_index=True)
        print(f"Combined {len(all_stations)} stations into single dataset")
        
        # Apply AQ-specific preprocessing
        df = self._apply_aq_preprocessing(df)
        
        return df
    
    def _apply_aq_preprocessing(self, df):
        """
        Apply AQ-specific preprocessing based on dataset description.
        """
        print("Applying AQ-specific preprocessing...")
        
        # Handle missing values (13.9% missing in worst case)
        missing_ratio = df.isnull().sum() / len(df)
        print(f"Missing value ratios: {missing_ratio[missing_ratio > 0].head().to_dict()}")
        
        # For AQ data, interpolate missing values for continuous variables
        continuous_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        for col in continuous_cols:
            if col in df.columns:
                # Group by station and interpolate within each station
                df[col] = df.groupby('station')[col].transform(lambda x: x.interpolate(method='linear'))
        
        # Handle categorical variables (fill with mode)
        categorical_cols = ['wd']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df.groupby('station')[col].transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'unknown'))
        
        # Validate temperature and precipitation values using helper method
        df = self._validate_and_clean_numeric_column(df, 'TEMP', min_val=-50, max_val=50)
        df = self._validate_and_clean_numeric_column(df, 'RAIN', min_val=0, max_val=100)
        df = self._validate_and_clean_numeric_column(df, 'WSPM', min_val=0, max_val=50)  # Wind speed validation
        
        # Remove any remaining rows with excessive missing values
        missing_threshold = 0.5  # Remove rows with >50% missing values
        valid_mask = df.isnull().sum(axis=1) / len(df.columns) < missing_threshold
        if (~valid_mask).any():
            df = df[valid_mask]
            print(f"Removed {(~valid_mask).sum()} rows with excessive missing values")
        
        return df.reset_index(drop=True)
    
    def _load_traffic_data(self):
        """
        Load traffic dataset from Metro_Interstate_Traffic_Volume.csv file.
        """
        source_config = self.config['dataset']['source']
        file_path = source_config['file_path']
        
        print(f"Loading traffic data from: {file_path}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Convert date_time to datetime
        df['date_time'] = pd.to_datetime(df['date_time'])
        
        print(f"Original traffic data shape: {df.shape}")
        print(f"Time range: {df['date_time'].min()} to {df['date_time'].max()}")
        
        # Apply traffic-specific preprocessing
        df = self._apply_traffic_preprocessing(df)
        
        return df
    
    def _apply_traffic_preprocessing(self, df):
        """
        Apply traffic-specific preprocessing.
        """
        traffic_config = self.config.get('preprocessing', {}).get('traffic_preprocessing', {})
        
        # Convert temperature from Kelvin to Celsius with validation
        if traffic_config.get('temp_celsius', True):
            df = self._validate_and_clean_numeric_column(df, 'temp', min_val=0, max_val=400)
            df['temp'] = df['temp'] - 273.15
            df = self._validate_and_clean_numeric_column(df, 'temp', min_val=-40, max_val=50)
            
            print("Converted temperature from Kelvin to Celsius")
        
        if traffic_config.get('cap_extreme_rain', True):
            extreme_threshold = traffic_config.get('extreme_rain_threshold', 50.0)
            df = self._validate_and_clean_numeric_column(df, 'rain_1h', min_val=0, max_val=extreme_threshold)
            df = self._validate_and_clean_numeric_column(df, 'snow_1h', min_val=0, max_val=10.0)  # Snow validation
        
        # Apply log transformation to traffic volume
        if traffic_config.get('log_transform', True):
            df['traffic_volume'] = np.log1p(df['traffic_volume'])
            print("Applied log transformation to traffic volume")
        
        # Handle outliers
        outlier_threshold = traffic_config.get('outlier_threshold', 3.0)
        if outlier_threshold > 0:
            # Apply outlier detection to traffic volume
            mean_traffic = df['traffic_volume'].mean()
            std_traffic = df['traffic_volume'].std()
            
            lower_bound = mean_traffic - outlier_threshold * std_traffic
            upper_bound = mean_traffic + outlier_threshold * std_traffic
            
            outlier_mask = (df['traffic_volume'] < lower_bound) | (df['traffic_volume'] > upper_bound)
            if outlier_mask.any():
                df.loc[outlier_mask, 'traffic_volume'] = np.clip(
                    df.loc[outlier_mask, 'traffic_volume'], 
                    lower_bound, 
                    upper_bound
                )
                print(f"Clipped {outlier_mask.sum()} outliers in traffic volume")
        
        return df
    
    def _generate_synthetic_harmonic_data(self):
        """
        Generate synthetic harmonic oscillator data with variable mass.
        """
        def compute_acceleration(x, v, t, gamma, k, m0, alpha):
            m_t = m0 + alpha * t
            a = - (gamma * v + k * x) / m_t
            return a  

        def sample_mixed_alpha(current_split, all_splits, sampling_ratios, n_samples):
            """Sample alpha values with mixed ratios from different splits"""
            main_ratio = sampling_ratios.get('main', 0.8)
            cross_ratio = sampling_ratios.get('cross', 0.2)
            
            # Number of samples from each source
            n_main = int(n_samples * main_ratio)
            n_cross_total = n_samples - n_main
            
            # Get other splits
            other_splits = [split_name for split_name in all_splits.keys() if split_name != current_split]
            n_cross_each = n_cross_total // len(other_splits) if other_splits else 0
            n_cross_remainder = n_cross_total % len(other_splits) if other_splits else 0
            
            alpha_samples = []
            
            # Sample from main range
            main_range = all_splits[current_split]['alpha']
            alpha_main = np.random.uniform(main_range[0], main_range[1], n_main)
            alpha_samples.extend(alpha_main)
            print(f"  - {n_main} samples from main range {main_range}")
            
            # Sample from cross ranges
            for i, other_split in enumerate(other_splits):
                other_range = all_splits[other_split]['alpha']
                n_cross_this = n_cross_each + (1 if i < n_cross_remainder else 0)
                if n_cross_this > 0:
                    alpha_cross = np.random.uniform(other_range[0], other_range[1], n_cross_this)
                    alpha_samples.extend(alpha_cross)
                    print(f"  - {n_cross_this} samples from {other_split} range {other_range}")
            
            # Shuffle to mix the samples
            np.random.shuffle(alpha_samples)
            return alpha_samples
        
        def generate_split_dataset(current_split, split_config, all_splits, sampling_ratios, physics_params):
            """Generate dataset for a specific split with mixed parameter sampling"""
            n_samples = split_config['n_samples']
            
            x_sequences = []
            c_sequences = []
            alpha_all = []
            
            # Get mixed parameter samples
            alpha_samples = sample_mixed_alpha(current_split, all_splits, sampling_ratios, n_samples)
            
            for alpha in alpha_samples:
                x_seq, v, a = generate_trajectory(
                    alpha, 
                    T=physics_params['T'],
                    steps=physics_params['steps'],
                    m0=physics_params['m0'],
                    gamma=physics_params['gamma'],
                    k=physics_params['k']
                )
                
                c_seq = np.column_stack([x_seq, v]) 
                
                x_sequences.append(a)
                c_sequences.append(c_seq)
                alpha_all.append(alpha)
            
            return np.array(x_sequences), np.array(c_sequences), np.array(alpha_all)
        
        print("Generating synthetic harmonic oscillator data...")
        
        # Get configuration
        physics_params = self.config['physics_params']
        
        # Check if this is variable parameters (VP) version
        is_vp = self.config['dataset'].get('name') == 'harmonic_vp'
        
        if is_vp:
            # For VP: use combination-based generation with multiple parameters
            combination_config = self.config['combination_based']
            
            def harmonic_oscillator_with_variable_para(t, y, alpha, beta, eta, m0=1.0, gamma0=0.1, k0=1.0, omega_m=0.1, omega_gamma=0.2, lambda_k=0.05):
                """Harmonic oscillator with variable mass, damping, and spring constant"""
                x, v = y
                # Time-varying parameters
                m_t = m0 + alpha * t
                gamma_t = gamma0 + beta * np.sin(omega_gamma * t)
                k_t = k0 * (1 - eta * (1 - np.exp(-lambda_k * t)))
                
                dxdt = v
                dvdt = -(gamma_t * v + k_t * x) / m_t
                return [dxdt, dvdt]
            
            def generate_vp_trajectory(alpha, beta, eta, physics_params):
                """Generate trajectory for VP version"""
                T = physics_params.get('t_max', 10.0)
                steps = physics_params.get('seq_len', 96)
                t_eval = np.linspace(0, T, steps)
                # Random sampling of initial conditions
                x0_range = physics_params.get('x0_range', [0.5, 1.5])
                v0_range = physics_params.get('v0_range', [-0.5, 0.5])
                x0 = np.random.uniform(x0_range[0], x0_range[1])
                v0 = np.random.uniform(v0_range[0], v0_range[1])
                y0 = [x0, v0]
                sol = solve_ivp(
                    harmonic_oscillator_with_variable_para,
                    [0, T], y0, t_eval=t_eval,
                    args=(alpha, beta, eta, physics_params.get('m0', 1.0), 
                          physics_params.get('gamma0', 0.1), physics_params.get('k0', 1.0),
                          physics_params.get('omega_m', 0.1), physics_params.get('omega_gamma', 0.2),
                          physics_params.get('lambda_k', 0.05))
                )
                x = sol.y[0]
                v = sol.y[1]
                # Compute acceleration for VP
                a = []
                for i, time in enumerate(t_eval):
                    m_t = physics_params.get('m0', 1.0) + alpha * np.tanh(physics_params.get('omega_m', 0.1) * time)
                    gamma_t = physics_params.get('gamma0', 0.1) + beta * np.sin(physics_params.get('omega_gamma', 0.2) * time)
                    k_t = physics_params.get('k0', 1.0) + eta * np.exp(-physics_params.get('lambda_k', 0.05) * time)
                    a_t = -(gamma_t * v[i] + k_t * x[i]) / m_t
                    a.append(a_t)
                return x, v, np.array(a)
            
            def sample_mixed_vp_parameters(current_split, all_splits, sampling_ratios, n_samples):
                """Sample VP parameters (alpha, beta, eta) with mixed ratios from different splits"""
                main_ratio = sampling_ratios.get('main', 0.8)
                cross_ratio = sampling_ratios.get('cross', 0.2)
                
                # Number of samples from each source
                n_main = int(n_samples * main_ratio)
                n_cross_total = n_samples - n_main
                
                # Get other splits
                other_splits = [split_name for split_name in all_splits.keys() if split_name != current_split]
                n_cross_each = n_cross_total // len(other_splits) if other_splits else 0
                n_cross_remainder = n_cross_total % len(other_splits) if other_splits else 0
                
                all_params = []
                
                # Sample from main range
                main_config = all_splits[current_split]
                for _ in range(n_main):
                    alpha = np.random.uniform(main_config['alpha'][0], main_config['alpha'][1])
                    beta = np.random.uniform(main_config['beta'][0], main_config['beta'][1])
                    eta = np.random.uniform(main_config['eta'][0], main_config['eta'][1])
                    all_params.append((alpha, beta, eta))
                    
                print(f"  - {n_main} samples from main ranges:")
                print(f"    α: {main_config['alpha']}, β: {main_config['beta']}, η: {main_config['eta']}")
                
                # Sample from cross ranges
                for i, other_split in enumerate(other_splits):
                    other_config = all_splits[other_split]
                    n_cross_this = n_cross_each + (1 if i < n_cross_remainder else 0)
                    if n_cross_this > 0:
                        for _ in range(n_cross_this):
                            alpha = np.random.uniform(other_config['alpha'][0], other_config['alpha'][1])
                            beta = np.random.uniform(other_config['beta'][0], other_config['beta'][1])
                            eta = np.random.uniform(other_config['eta'][0], other_config['eta'][1])
                            all_params.append((alpha, beta, eta))
                        print(f"  - {n_cross_this} samples from {other_split} ranges:")
                        print(f"    α: {other_config['alpha']}, β: {other_config['beta']}, η: {other_config['eta']}")
                
                # Shuffle to mix the samples
                np.random.shuffle(all_params)
                return all_params
            
            # Generate VP data with combination-based splitting and mixed sampling
            sampling_ratios = combination_config.get('sampling_ratios', {'main': 1.0, 'cross': 0.0})
            all_splits = combination_config['splits']
            all_sequences = []
            
            for split_name in ['train', 'val', 'test']:
                print(f"Generating {split_name} data with mixed sampling...")
                split_config = combination_config['splits'][split_name]
                n_samples = split_config['n_samples']
                
                # Get mixed parameter samples
                param_samples = sample_mixed_vp_parameters(split_name, all_splits, sampling_ratios, n_samples)
                
                for alpha, beta, eta in param_samples:
                    x_seq, v_seq, a_seq = generate_vp_trajectory(
                        alpha, beta, eta, physics_params
                    )
                    
                    # Conditioning is still [velocity, acceleration]
                    c_seq = np.column_stack([x_seq, v_seq])
                    
                    sequence_data = {
                        'x_sequence': a_seq,
                        'c_sequence': c_seq,
                        'alpha': alpha, 'beta': beta, 'eta': eta,
                        'split': split_name
                    }
                    all_sequences.append(sequence_data)
                
                print(f"Generated {n_samples} sequences for {split_name}")
            
            print(f"Total VP sequences generated: {len(all_sequences)}")
        else:
            # generate data for each split and store sequences directly
            def harmonic_oscillator_with_variable_mass(t, y, alpha, m0=1.0, gamma=0.1, k=1.0):
                """Harmonic oscillator with variable mass: m = m0 + alpha * t"""
                x, v = y
                m = m0 + alpha * t
                dxdt = v
                dvdt = -(gamma * v + k * x) / m
                return [dxdt, dvdt]
            
            def generate_trajectory(alpha, T=10.0, steps=100, m0=1.0, gamma=0.1, k=1.0):
                """Generate a single trajectory for given alpha"""
                t_eval = np.linspace(0, T, steps)
                
                # Random sampling of initial conditions
                x0_range = physics_params.get('x0_range', [0.5, 1.5])
                v0_range = physics_params.get('v0_range', [-0.5, 0.5])
                x0 = np.random.uniform(x0_range[0], x0_range[1])
                v0 = np.random.uniform(v0_range[0], v0_range[1])
                y0 = [x0, v0]
                sol = solve_ivp(
                    harmonic_oscillator_with_variable_mass, 
                    [0, T], y0, t_eval=t_eval, 
                    args=(alpha, m0, gamma, k)
                )
                x = sol.y[0]  # (steps,)
                v = sol.y[1]  # (steps,)
                a = compute_acceleration(x, v, t_eval, gamma, k, m0, alpha)
                return x, v, a
                # return a, x, v
        
            alpha_config = self.config['alpha_based']
            sampling_ratios = alpha_config.get('sampling_ratios', {'main': 1.0, 'cross': 0.0})
            all_splits = alpha_config['splits']
            all_sequences = []
            
            for split_name in ['train', 'val', 'test']:
                print(f"Generating {split_name} data with mixed sampling...")
                split_config = alpha_config['splits'][split_name]
                
                x_sequences, c_sequences, alpha_data = generate_split_dataset(
                    split_name, split_config, all_splits, sampling_ratios, physics_params
                )
                
                for i in range(len(x_sequences)):
                    sequence_data = {
                        'x_sequence': x_sequences[i],  
                        'c_sequence': c_sequences[i],    
                        'alpha': alpha_data[i],
                        'split': split_name
                    }
                    all_sequences.append(sequence_data)
            
            print(f"Generated {len(x_sequences)} sequences for {split_name} (alpha range: {split_config['alpha']})")
        
        print(f"Total synthetic sequences generated: {len(all_sequences)}")
        
        self.synthetic_sequences = all_sequences
        df = pd.DataFrame([
            {'split': seq['split'], 'alpha': seq['alpha']} 
            for seq in all_sequences
        ])
        
        return df
    
    def _generate_counterfactual_data(self):
        """
        Generate counterfactual data for test split.
        Keep environment parameters (alpha, beta, eta) the same but change initial conditions.
        """
        print("Generating counterfactual data for test split...")
        
        # Get test sequences 
        test_sequences = [seq for seq in self.synthetic_sequences if seq['split'] == 'test']
        
        # Check if this is VP or VM dataset
        is_vp = self.config['dataset'].get('name') == 'harmonic_vp'
        physics_params = self.config['physics_params']
        
        x_cf_list = []
        c_cf_list = []
        
        for seq in test_sequences:
            if is_vp:
                alpha = seq['alpha']
                beta = seq['beta'] 
                eta = seq['eta']
                
                x0_cf_range = physics_params.get('x0_cf_range', [1.2, 2.0])
                v0_cf_range = physics_params.get('v0_cf_range', [0.2, 0.8])
                x0_alt = np.random.uniform(x0_cf_range[0], x0_cf_range[1])
                v0_alt = np.random.uniform(v0_cf_range[0], v0_cf_range[1])
                
                x_cf_seq, v_cf_seq, a_cf_seq = self._generate_vp_trajectory_with_init(
                    alpha, beta, eta, physics_params, x0_alt, v0_alt
                )
                
            else:
                alpha = seq['alpha']
                
                x0_cf_range = physics_params.get('x0_cf_range', [1.2, 2.0])
                v0_cf_range = physics_params.get('v0_cf_range', [0.2, 0.8])
                x0_alt = np.random.uniform(x0_cf_range[0], x0_cf_range[1])
                v0_alt = np.random.uniform(v0_cf_range[0], v0_cf_range[1])
                
                x_cf_seq, v_cf_seq, a_cf_seq = self._generate_vm_trajectory_with_init(
                    alpha, physics_params, x0_alt, v0_alt
                )
            
            # Create condition sequence [velocity, acceleration]
            c_cf_seq = np.column_stack([x_cf_seq, v_cf_seq])
            
            x_cf_list.append(a_cf_seq.reshape(-1, 1))  # (96, 1)
            c_cf_list.append(c_cf_seq)  # (96, 2)
        
        # Convert to numpy arrays
        x_cf = np.array(x_cf_list)  # (n_test, 96, 1)
        c_cf = np.array(c_cf_list)  # (n_test, 96, 2)
        
        print(f"Generated {len(test_sequences)} counterfactual sequences")
        return x_cf, c_cf
    
    def _generate_vp_trajectory_with_init(self, alpha, beta, eta, physics_params, x0, v0):
        """Generate VP trajectory with specific initial conditions"""
        def harmonic_oscillator_with_variable_para(t, y, alpha, beta, eta, m0=1.0, gamma0=0.1, k0=1.0, omega_m=0.1, omega_gamma=0.2, lambda_k=0.05):
            x, v = y
            m_t = m0 + alpha * np.tanh(omega_m * t)
            gamma_t = gamma0 + beta * np.sin(omega_gamma * t)
            k_t = k0 + eta * np.exp(-lambda_k * t)
            dxdt = v
            dvdt = -(gamma_t * v + k_t * x) / m_t
            return [dxdt, dvdt]
        
        T = physics_params.get('t_max', 10.0)
        steps = physics_params.get('seq_len', 96)
        t_eval = np.linspace(0, T, steps)
        y0 = [x0, v0]
        
        sol = solve_ivp(
            harmonic_oscillator_with_variable_para, 
            [0, T], y0, t_eval=t_eval,
            args=(alpha, beta, eta, physics_params.get('m0', 1.0), 
                  physics_params.get('gamma0', 0.1), physics_params.get('k0', 1.0),
                  physics_params.get('omega_m', 0.1), physics_params.get('omega_gamma', 0.2),
                  physics_params.get('lambda_k', 0.05))
        )
        
        x = sol.y[0]
        v = sol.y[1]
        
        # Compute acceleration
        a = []
        for i, time in enumerate(t_eval):
            m_t = physics_params.get('m0', 1.0) + alpha * np.tanh(physics_params.get('omega_m', 0.1) * time)
            gamma_t = physics_params.get('gamma0', 0.1) + beta * np.sin(physics_params.get('omega_gamma', 0.2) * time)
            k_t = physics_params.get('k0', 1.0) + eta * np.exp(-physics_params.get('lambda_k', 0.05) * time)
            a_t = -(gamma_t * v[i] + k_t * x[i]) / m_t
            a.append(a_t)
        
        return x, v, np.array(a)
    
    def _generate_vm_trajectory_with_init(self, alpha, physics_params, x0, v0):
        """Generate VM trajectory with specific initial conditions"""
        def harmonic_oscillator_with_variable_mass(t, y, alpha, m0=1.0, gamma=0.1, k=1.0):
            x, v = y
            m = m0 + alpha * t
            dxdt = v
            dvdt = -(gamma * v + k * x) / m
            return [dxdt, dvdt]
        
        def compute_acceleration(x, v, t, gamma, k, m0, alpha):
            m_t = m0 + alpha * t  
            a = - (gamma * v + k * x) / m_t
            return a
        
        T = physics_params['T']
        steps = physics_params['steps']
        t_eval = np.linspace(0, T, steps)
        y0 = [x0, v0]
        
        sol = solve_ivp(
            harmonic_oscillator_with_variable_mass, 
            [0, T], y0, t_eval=t_eval, 
            args=(alpha, physics_params['m0'], physics_params['gamma'], physics_params['k'])
        )
        
        x = sol.y[0]  
        v = sol.y[1]  
        a = compute_acceleration(x, v, t_eval, physics_params['gamma'], physics_params['k'], 
                               physics_params['m0'], alpha)
        
        return x, v, a
        # return a, x, v
    
    def load_and_preprocess_data(self):
        """
        Load raw data and perform initial preprocessing.
        Supports multi-station, single file data sources.
        
        Returns:
        --------
        df : pandas.DataFrame
            Preprocessed dataframe
        """
        # Check source type and load accordingly
        if 'source' in self.config['dataset']:
            source_type = self.config['dataset']['source'].get('type', 'multi_station')
            if source_type == 'traffic':
                df = self._load_traffic_data()
            elif source_type == 'synthetic':
                df = self._generate_synthetic_harmonic_data()
            elif source_type == 'synthetic_vp':
                df = self._generate_synthetic_harmonic_data()
            else:
                df = self._load_multi_station_data()
        else:
            df = self._generate_synthetic_harmonic_data()
        
        print(f"Loaded dataset: {self.config['dataset']['name']}")
        print(f"Original shape: {df.shape}")
        
        # Handle datetime if present
        datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if datetime_cols and len(datetime_cols) == 1:
            datetime_col = datetime_cols[0]
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        elif 'year' in df.columns and 'month' in df.columns and 'day' in df.columns and 'hour' in df.columns:
            # Create datetime from separate columns
            df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        
        # Sort by datetime
        if 'datetime' in df.columns:
            df = df.sort_values('datetime').reset_index(drop=True)
        
        # Handle missing values
        preprocessing_config = self.config.get('preprocessing', {})
        if preprocessing_config.get('handle_missing', False):
            df = self._handle_missing_values(df, preprocessing_config)
        
        # Add temporal features
        if preprocessing_config.get('add_temporal_features', False):
            df = self._add_temporal_features(df)
            print("Added temporal features")
        
        # Encode categorical features
        df = self._encode_categorical_features(df)
        
        print(f"Processed shape: {df.shape}")
        self.df = df
        return df
    
    def _handle_missing_values(self, df, config):
        """Handle missing values in the dataset."""
        missing_method = config.get('missing_method', 'interpolate')
        max_missing_ratio = config.get('max_missing_ratio', 0.1)
        
        # Check missing values
        missing_ratio = df.isnull().sum() / len(df)
        print(f"Missing value ratios: {missing_ratio[missing_ratio > 0].to_dict()}")
        
        # Drop columns with too many missing values
        cols_to_drop = missing_ratio[missing_ratio > max_missing_ratio].index
        if len(cols_to_drop) > 0:
            print(f"Dropping columns with >{max_missing_ratio*100}% missing: {list(cols_to_drop)}")
            df = df.drop(columns=cols_to_drop)
        
        # Handle remaining missing values
        if missing_method == 'interpolate':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        elif missing_method == 'forward_fill':
            df = df.fillna(method='ffill')
        elif missing_method == 'fill_value':
            print(f"Will fill missing values after normalization")
        elif missing_method == 'drop':
            df = df.dropna()
        
        # Final cleanup for interpolate and forward_fill methods
        if missing_method not in ['fill_value', 'drop']:
            df = df.dropna()  # Drop any remaining NaN rows
        
        return df
    
    def _add_temporal_features(self, df):
        """Add temporal features for time series analysis."""
        if 'datetime' in df.columns:
            df['hour'] = df['datetime'].dt.hour
            df['day'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            df['year'] = df['datetime'].dt.year
            df['weekday'] = df['datetime'].dt.weekday
            df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        
        # Cyclical encoding for hour and month
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['hour_encoded'] = df['hour'] / 24  # Normalized hour
        
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Season encoding
        if 'month' in df.columns:
            def get_season(month):
                if month in [12, 1, 2]:
                    return 'winter'
                elif month in [3, 4, 5]:
                    return 'spring'
                elif month in [6, 7, 8]:
                    return 'summer'
                else:
                    return 'autumn'
            
            df['season'] = df['month'].apply(get_season)
        
        # Wind direction grouping (if exists)
        if 'wd' in df.columns:
            def group_wind_direction(wd):
                if pd.isna(wd):
                    return 'unknown'
                direction_groups = {
                    'N': ['N', 'NNE', 'NNW'],
                    'E': ['E', 'ENE', 'ESE'], 
                    'S': ['S', 'SSE', 'SSW'],
                    'W': ['W', 'WSW', 'WNW'],
                    'NE': ['NE'],
                    'SE': ['SE'],
                    'SW': ['SW'],
                    'NW': ['NW']
                }
                
                for group, directions in direction_groups.items():
                    if wd in directions:
                        return group
                return 'unknown'
            
            df['wind_direction_group'] = df['wd'].apply(group_wind_direction)
        
        return df
    
    def _generate_dataset_info(self, data_splits, output_dir):
        """
        Generate a text file with dataset information.
        """
        dataset_name = self.config['dataset']['name']
        info_path = f"{output_dir}/{dataset_name}_info.txt"
        
        # Get configuration info
        preprocessing_config = self.config.get('preprocessing', {})
        split_method = preprocessing_config.get('split_method', 'ratio')
        seq_len = preprocessing_config.get('seq_len', 96)
        missing_method = preprocessing_config.get('missing_method', 'interpolate')
        missing_fill_value = preprocessing_config.get('missing_fill_value', -1)
        
        # Get variable info
        variables = self.config.get('variables', {})
        target_var = variables.get('x_var', 'Unknown')
        
        # Get encoding info
        encoding_info = []
        categorical_vars = self.config.get('feature_encoding', {}).get('categorical_vars', {})
        for var, config in categorical_vars.items():
            if config.get('type') == 'embedding':
                encoding_info.append(f"  {var}: embedding (dim={config.get('embedding_dim', 8)}, categories={len(config.get('categories', []))}+1)")
        
        # Create info content
        info_content = f"""Dataset Information: {dataset_name.upper()} 
{'='*50}

Dataset Shapes:
  Training:   x={data_splits['x_train'].shape}, c={data_splits['c_train'].shape}
  Validation: x={data_splits['x_val'].shape}, c={data_splits['c_val'].shape}
  Testing:    x={data_splits['x_test'].shape}, c={data_splits['c_test'].shape}
  Total:      {data_splits['x_train'].shape[0] + data_splits['x_val'].shape[0] + data_splits['x_test'].shape[0]} sequences

Variables:
  Target (x):     {target_var}
  Conditions (c): {self.c_var_names} ({len(self.c_var_names)} variables)

Configuration:
  Sequence Length: {seq_len}
  Split Method:    {split_method}
  Missing Method:  {missing_method}
  Fill Value:      {missing_fill_value}

Feature Encoding:
{chr(10).join(encoding_info) if encoding_info else '  None'}

Split Details:
"""
        # Add split-specific details
        if split_method == 'station_based':
            station_config = self.config.get('station_based', {})
            info_content += f"  Train Stations: {station_config.get('train', [])}\n"
            info_content += f"  Val Stations:   {station_config.get('val', [])}\n"
            info_content += f"  Test Stations:  {station_config.get('test', [])}\n"
        elif split_method == 'client_based':
            client_config = self.config.get('client_based', {})
            info_content += f"  Train Clients: {client_config.get('train', [])}\n"
            info_content += f"  Val Clients:   {client_config.get('val', [])}\n"
            info_content += f"  Test Clients:  {client_config.get('test', [])}\n"
        elif split_method == 'holiday_based':
            holiday_config = self.config.get('holiday_based', {})
            info_content += f"  Train: Non-holiday ({holiday_config.get('train_ratio', 0.67):.0%})\n"
            info_content += f"  Val: Non-holiday ({holiday_config.get('val_ratio', 0.33):.0%})\n"
            info_content += f"  Test: Holiday records\n"
        elif split_method == 'temp_based':
            temp_config = self.config.get('temp_based', {})
            info_content += f"  Train: Cold ({temp_config.get('train_temp_range', [-50, 15])}°C)\n"
            info_content += f"  Val: Moderate ({temp_config.get('val_temp_range', [15, 25])}°C)\n"
            info_content += f"  Test: Hot ({temp_config.get('test_temp_range', [25, 50])}°C)\n"
        elif split_method == 'year_based':
            time_config = self.config.get('year_based', {})
            info_content += f"  Train Years: {time_config.get('train', [])}\n"
            info_content += f"  Val Years:   {time_config.get('val', [])}\n"
            info_content += f"  Test Years:  {time_config.get('test', [])}\n"
        elif split_method == 'season_based':
            season_config = self.config.get('season_based', {})
            info_content += f"  Train Seasons: {season_config.get('train', [])}\n"
            info_content += f"  Val Seasons:   {season_config.get('val', [])}\n"
            info_content += f"  Test Seasons:  {season_config.get('test', [])}\n"
        
        categorical_vars = self.c_cat_names
        binary_vars = self.c_binary_names  
        continuous_vars = self.c_continuous_names
        
        continuous_indices = list(self.continuous_indices)
        c_train = data_splits['c_train']
        
        if continuous_indices:
            continuous_data = c_train[:, :, continuous_indices]
            continuous_min = continuous_data.min()
            continuous_max = continuous_data.max()
            continuous_range_text = f"[{continuous_min:.3f}, {continuous_max:.3f}]"
        else:
            continuous_range_text = "No continuous variables"
        
        info_content += f"""
Data Quality:
  Missing Values: Filled with {missing_fill_value} after normalization
  Normalization:  StandardScaler applied to continuous features only
  
Variable Types:
  Continuous ({len(continuous_vars)}): {', '.join(continuous_vars) if continuous_vars else 'None'}
  Categorical ({len(categorical_vars)}): {', '.join(categorical_vars) if categorical_vars else 'None'} 
  Binary ({len(binary_vars)}): {', '.join(binary_vars) if binary_vars else 'None'}

Data Ranges:
  Target: [{data_splits['x_train'].min():.3f}, {data_splits['x_train'].max():.3f}]
  Continuous Variables: {continuous_range_text}
  Binary Variables: {{0, 1}}

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Write info file
        with open(info_path, 'w') as f:
            f.write(info_content)
        
        print(f"Dataset info saved to: {info_path}")
        return info_path
    
    def _encode_categorical_features(self, df):
        """Encode categorical features according to configuration."""
        legacy_encoding_config = self.config.get('feature_encoding', {})
        legacy_categorical_vars = legacy_encoding_config.get('categorical_vars', {})
        feature_embeddings = self.config.get('feature_embeddings', {})
        all_categorical_vars = {**legacy_categorical_vars, **feature_embeddings}
        
        for var_name, encoding_info in all_categorical_vars.items():
            categories = encoding_info.get('categories', [])
            if categories:
                # Create mapping from category to index
                cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
                cat_to_idx['unknown'] = len(categories)  # Unknown category
                
                # Map values to indices
                df[f'{var_name}_encoded'] = df[var_name].map(cat_to_idx).fillna(len(categories))
                print(f"Encoded {var_name} for embedding: {len(categories)} categories + unknown")
                
                # Store encoding info for later use
                self.encoders[f'{var_name}_embedding'] = {
                    'categories': categories,
                    'embedding_dim': encoding_info.get('embedding_dim', 8),
                    'vocab_size': len(categories) + 1  # +1 for unknown
                }
                

        if 'holiday' in df.columns and 'holiday' not in all_categorical_vars:
            # Binary encoding: None -> 0, Any other value -> 1
            df['holiday_encoded'] = (df['holiday'] != 'None').astype(int)
            print(f"Binary encoded holiday: None->0, Others->1")
            
            # Store encoding info
            self.encoders['holiday_binary'] = {
                'type': 'binary',
                'mapping': {'None': 0, 'Holiday': 1}
            }
        
        return df
    
    def create_sequences(self, df, seq_len=96, interval=1):
        """
        Create sequences from time series data with improved efficiency.
        """
        # For synthetic data, the data is already in the correct format - no sequence creation needed
        if self.config['dataset']['source'].get('type') == 'synthetic':
            print("Synthetic data detected - using direct format (no sequence creation needed)")
            return [df]  # Return as a single "sequence" that contains all data
        
        sequences = []
        
        # Validation config
        min_valid_ratio =  0.8
        
        # Pre-calculate null counts for efficiency
        null_counts = df.isnull().sum(axis=1) if not df.empty else pd.Series()
        
        # For multi-station data, create sequences within each station to maintain temporal continuity
        if 'station' in df.columns:
            print("Creating sequences by station for temporal continuity...")
            # Group by station once for efficiency
            station_groups = df.groupby('station')
            
            for station, station_data in station_groups:
                station_data = station_data.sort_values('datetime').reset_index(drop=True)
                data_len = len(station_data)
                
                if data_len < seq_len:
                    continue
                
                # Generate all possible sequences using the specified interval
                for start_idx in range(0, data_len - seq_len + 1, interval):
                    end_idx = start_idx + seq_len
                    
                    # Quick validation check using pre-calculated null counts
                    seq_null_count = null_counts.iloc[start_idx:end_idx].sum()
                    total_cells = seq_len * len(df.columns)
                    valid_ratio = 1 - (seq_null_count / total_cells)
                    
                    if valid_ratio >= min_valid_ratio:
                        sequence = station_data.iloc[start_idx:end_idx].copy()
                        sequence['sequence_id'] = len(sequences)
                        sequences.append(sequence)
        else:
            # Optimized logic for non-station data
            data_len = len(df)
            for start_idx in range(0, data_len - seq_len + 1, interval):
                end_idx = start_idx + seq_len
                
                # Quick validation check using pre-calculated null counts
                seq_null_count = null_counts.iloc[start_idx:end_idx].sum()
                total_cells = seq_len * len(df.columns)
                valid_ratio = 1 - (seq_null_count / total_cells)
                
                if valid_ratio >= min_valid_ratio:
                    sequence = df.iloc[start_idx:end_idx].copy()
                    sequence['sequence_id'] = len(sequences)
                    sequences.append(sequence)
        
        print(f"Created {len(sequences)} sequences of length {seq_len}")
        return sequences
    
    def extract_tsg_features(self, sequences):
        """
        Extract features for TSG tasks (x and c) with improved efficiency.
        """
        if self.config['dataset']['source'].get('type') in ['synthetic', 'synthetic_vp']:
            return self._extract_synthetic_features(sequences[0])  # sequences[0] is the full DataFrame
        
        variables_config = self.config.get('variables', {})
        target_var = variables_config.get('x_var', None)
        
        num_sequences = len(sequences)
        if num_sequences == 0:
            return np.empty((0, 0, 1)), np.empty((0, 0, 0)), []
        
        seq_len = len(sequences[0])
        num_conditions = len(self.c_var_names)
        
        # Initialize arrays
        x_data = np.empty((num_sequences, seq_len, 1), dtype=np.float32)
        c_data = np.empty((num_sequences, seq_len, num_conditions), dtype=np.float32)
        metadata = []
        
        # Process sequences efficiently
        valid_sequences = 0
        for i, seq in enumerate(sequences):
            # Target sequence (x)
            
            x_data[valid_sequences, :, 0] = seq[target_var].values
            
            # Condition sequences (c) - optimized processing
            for j, var in enumerate(self.c_var_names):
                # Check if this variable has an encoded version
                encoded_var = f'{var}_encoded'
                if encoded_var in seq.columns:
                    c_data[valid_sequences, :, j] = seq[encoded_var].values
                elif var in seq.columns:
                    c_data[valid_sequences, :, j] = seq[var].values
            
            # Metadata - optimized extraction
            metadata.append(self._extract_sequence_metadata(seq, i))
            valid_sequences += 1
        
        # Trim arrays to valid sequences
        x_data = x_data[:valid_sequences]
        c_data = c_data[:valid_sequences]
        
        print(f"Extracted features: x_data {x_data.shape}, c_data {c_data.shape}")
        
        self.sequences_metadata = metadata
        
        return x_data, c_data, metadata
    
    def _extract_synthetic_features(self, df):
        """
        Extract features from synthetic data sequences.
        """
        # Use the stored sequences
        sequences = self.synthetic_sequences
        num_sequences = len(sequences)
        
        if num_sequences == 0:
            return np.empty((0, 0, 1)), np.empty((0, 0, 0)), []
        
        # Get sequence length from first sequence
        seq_len = len(sequences[0]['x_sequence'])
        num_conditions = sequences[0]['c_sequence'].shape[1]  # 2 for original, 3 for VM version
        
        # Initialize arrays with correct sequence dimensions
        x_data = np.empty((num_sequences, seq_len, 1), dtype=np.float32)
        c_data = np.empty((num_sequences, seq_len, num_conditions), dtype=np.float32)
        
        # Extract sequences
        metadata = []
        for i, seq in enumerate(sequences):
            x_data[i, :, 0] = seq['x_sequence']
            c_data[i, :, :] = seq['c_sequence']
            
            # Metadata - handle both original (alpha only) and VM (alpha, beta, eta) versions
            metadata_entry = {
                'index': i, 
                'alpha': seq['alpha'], 
                'split': seq['split']
            }
            if 'beta' in seq:
                metadata_entry['beta'] = seq['beta']
            if 'eta' in seq:
                metadata_entry['eta'] = seq['eta']
            
            metadata.append(metadata_entry)
        
        print(f"Extracted synthetic sequences: x_data {x_data.shape}, c_data {c_data.shape}")
        
        self.sequences_metadata = metadata
        
        return x_data, c_data, metadata
    
    def _get_condition_variables(self, split_method):
        split_config = self.config.get(split_method, {})
        return split_config['c_var']
        
    
    def _extract_sequence_metadata(self, seq, sequence_id):
        """
        Extract metadata from a sequence efficiently.
        """
        # Basic metadata
        meta = {
            'sequence_id': sequence_id,
            'start_time': seq.iloc[0].get('datetime', seq.iloc[0].get('date_time', None)),
            'end_time': seq.iloc[-1].get('datetime', seq.iloc[-1].get('date_time', None)),
            'station': seq.iloc[0].get('station', None),
            'client': seq.iloc[0].get('client', None)
        }
        
        # Traffic-specific metadata
        if 'holiday' in seq.columns:
            # Check if any record in the sequence is a holiday
            is_holiday = (seq['holiday'] != 'None').any()
            
            # Check if weekend should be treated as holiday
            if 'holiday_based' in self.config and self.config['holiday_based'].get('weekend_as_holiday', False):
                # Add weekend check if datetime column exists
                datetime_col = 'datetime' if 'datetime' in seq.columns else 'date_time' if 'date_time' in seq.columns else None
                if datetime_col:
                    is_weekend = (seq[datetime_col].dt.weekday >= 5).any()
                    is_holiday = is_holiday or is_weekend
            
            meta['holiday'] = is_holiday
        
        if 'temp' in seq.columns:
            meta['avg_temp'] = seq['temp'].mean()
        
        if 'weather_main' in seq.columns:
            weather_mode = seq['weather_main'].mode()
            meta['weather_main'] = weather_mode.iloc[0] if not weather_mode.empty else None
        
        return meta
    
    def normalize_data(self, x_data, c_data):
        """
        Normalize the data using StandardScaler with improved efficiency.
        Skip categorical features to preserve integer indices for embedding layers.
        """
        preprocessing_config = self.config.get('preprocessing', {})
        
        # Normalize target (x)
        if preprocessing_config.get('normalize_target', True):
            x_reshaped = x_data.reshape(-1, 1)
            self.scalers['target'] = StandardScaler()
            x_normalized = self.scalers['target'].fit_transform(x_reshaped)
            x_normalized = x_normalized.reshape(x_data.shape)
        else:
            x_normalized = x_data.copy()
        
        # Use pre-setup variable classifications (optimized)
        print(f"Condition variables order: {self.c_var_names}")
        print(f"Variable to index mapping: {self.var_to_index}")
        
        # Get categorical feature indices to skip normalization
        categorical_indices = self.cat_indices
        
        # Use pre-setup binary indices (no need to detect dynamically)
        binary_indices = self.binary_indices
        print(f"Pre-setup binary indices: {binary_indices}")
        
        # Combine categorical and binary variable indices to skip normalization
        skip_normalization_indices = categorical_indices.union(binary_indices)
        print(f"Skipping normalization for indices: {skip_normalization_indices}")
        
        # Normalize conditions (c) - skip categorical and binary features
        if preprocessing_config.get('normalize_conditions', True):
            N, T, C = c_data.shape
            c_normalized = c_data.copy().astype(np.float32)  # Preserve original data
            
            # Only normalize numerical features (skip categorical and binary features)
            for i in range(C):
                var_name = self.c_var_names[i] if i < len(self.c_var_names) else f"unknown_{i}"
                if i not in skip_normalization_indices:
                    # Normalize numerical features
                    c_slice = c_data[:, :, i]
                    c_reshaped = c_slice.reshape(-1, 1)
                    scaler = StandardScaler()
                    c_norm = scaler.fit_transform(c_reshaped)
                    c_normalized[:, :, i] = c_norm.reshape(N, T)
                    self.scalers[f'condition_{i}'] = scaler
                    print(f"Normalized '{var_name}' at index {i}")
                else:
                    # Keep categorical and binary features as original values
                    print(f"Skipping normalization for '{var_name}' at index {i} (categorical or binary)")
        else:
            c_normalized = c_data.copy()
        
        # Handle post-normalization missing value filling (only for numerical features)
        missing_method = preprocessing_config.get('missing_method', 'interpolate')
        
        if missing_method == 'fill_value':
            fill_value = preprocessing_config.get('missing_fill_value', -1)
            
            # Efficient NaN handling
            if np.isnan(x_normalized).any():
                x_normalized = np.nan_to_num(x_normalized, nan=fill_value)
                print(f"Filled missing values in target with: {fill_value}")
                
            # Only fill NaN for numerical features (skip categorical and binary)
            if np.isnan(c_normalized).any():
                for i in range(c_normalized.shape[2]):
                    var_name = self.c_var_names[i] if i < len(self.c_var_names) else f"unknown_{i}"
                    if i not in skip_normalization_indices and np.isnan(c_normalized[:, :, i]).any():
                        c_normalized[:, :, i] = np.nan_to_num(c_normalized[:, :, i], nan=fill_value)
                        print(f"Filled missing values in '{var_name}' (index {i}) with: {fill_value}")
        
        print("Data normalization completed (categorical features preserved as integers)")
        return x_normalized, c_normalized
    
    def split_data(self, x_data, c_data):
        preprocessing_config = self.config.get('preprocessing', {})
        split_method = preprocessing_config.get('split_method', None)
        
        # For synthetic datasets, auto-detect split method
        if split_method is None:
            source_type = self.config.get('dataset', {}).get('source', {}).get('type')
            if source_type == 'synthetic':
                split_method = 'alpha_based'
            elif source_type == 'synthetic_vp':
                split_method = 'combination_based'
            else:
                split_method = 'ratio'  # Default fallback
        
        if split_method == 'station_based' in self.config:
            # Use station-based splitting
            return self.split_data_by_station(x_data, c_data)
        elif split_method == 'temp_based':
            # Use temp-based splitting for traffic data
            return self.split_data_by_temp(x_data, c_data)
        elif split_method == 'alpha_based':
            # Use alpha-based splitting for synthetic data
            return self.split_data_by_alpha(x_data, c_data)
        elif split_method == 'combination_based':
            # Use combination-based splitting for multi-parameter synthetic data
            return self.split_data_by_combination(x_data, c_data)
    
    def split_data_by_station(self, x_data, c_data):
        """
        Split data based on station assignments.
        """
        station_config = self.config['station_based']
        train_stations = station_config['train']
        val_stations = station_config['val']
        test_stations = station_config['test']
        
        # Get station information from sequences
        if not hasattr(self, 'sequences_metadata'):
            raise ValueError("Station-based splitting requires sequence metadata with station information")
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for i, seq in enumerate(self.sequences_metadata):
            station = seq.get('station', None)
            if station in train_stations:
                train_indices.append(i)
            elif station in val_stations:
                val_indices.append(i)
            elif station in test_stations:
                test_indices.append(i)
            else:
                print(f"Warning: Station '{station}' not found in split configuration, assigning to train")
                train_indices.append(i)
        
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)
        
        # Create splits
        splits = {
            'x_train': x_data[train_indices],
            'x_val': x_data[val_indices], 
            'x_test': x_data[test_indices],
            'c_train': c_data[train_indices],
            'c_val': c_data[val_indices],
            'c_test': c_data[test_indices]
        }
        
        print(f"Station-based split - Train: {len(train_indices)} (stations: {train_stations})")
        print(f"                    - Val: {len(val_indices)} (stations: {val_stations})")
        print(f"                    - Test: {len(test_indices)} (stations: {test_stations})")
        
        return splits
    

    def split_data_by_temp(self, x_data, c_data):
        """
        Split data based on temperature conditions for traffic data.
        """
        temp_config = self.config['temp_based']
        train_temp_range = temp_config.get('train_temp_range')
        val_temp_range = temp_config.get('val_temp_range')
        test_temp_range = temp_config.get('test_temp_range')

        train_indices = []
        val_indices = []
        test_indices = []
        
        for i, seq in enumerate(self.sequences_metadata):
            # Get average temperature for the sequence
            avg_temp = seq.get('avg_temp', None)
            if train_temp_range[0] <= avg_temp < train_temp_range[1]:
                train_indices.append(i)
            elif val_temp_range[0] <= avg_temp < val_temp_range[1]:
                val_indices.append(i)
            elif test_temp_range[0] <= avg_temp < test_temp_range[1]:
                test_indices.append(i)
        
        train_indices = np.array(train_indices, dtype=int)
        val_indices = np.array(val_indices, dtype=int)
        test_indices = np.array(test_indices, dtype=int)
        
        # Create splits
        splits = {
            'x_train': x_data[train_indices],
            'x_val': x_data[val_indices], 
            'x_test': x_data[test_indices],
            'c_train': c_data[train_indices],
            'c_val': c_data[val_indices],
            'c_test': c_data[test_indices]
        }
        
        print(f"Temperature-based split - Train: {len(train_indices)} (cold: {train_temp_range[0]}-{train_temp_range[1]}°C)")
        print(f"                    - Val: {len(val_indices)} (moderate: {val_temp_range[0]}-{val_temp_range[1]}°C)")
        print(f"                    - Test: {len(test_indices)} (hot: {test_temp_range[0]}-{test_temp_range[1]}°C)")
        
        return splits
    
    
    def split_data_by_alpha(self, x_data, c_data):
        """
        Split synthetic data based on pre-assigned split column.
        The synthetic data generator already assigns train/val/test splits.
        """
        split_column = self.df['split'].values
        
        # Create masks for each split
        train_mask = split_column == 'train'
        val_mask = split_column == 'val'
        test_mask = split_column == 'test'
        
        splits = {
            'x_train': x_data[train_mask],
            'x_val': x_data[val_mask],
            'x_test': x_data[test_mask],
            'c_train': c_data[train_mask],
            'c_val': c_data[val_mask],
            'c_test': c_data[test_mask]
        }
        
        print(f"Alpha-based split - Train: {np.sum(train_mask)} samples")
        print(f"                  - Val: {np.sum(val_mask)} samples")
        print(f"                  - Test: {np.sum(test_mask)} samples")
        
        return splits
    
    def split_data_by_combination(self, x_data, c_data):
        """
        Split synthetic data with parameter combinations based on pre-assigned split column.
        For harmonic_vm dataset with alpha, beta, eta parameter combinations.
        """
        split_column = self.df['split'].values
        
        # Create masks for each split
        train_mask = split_column == 'train'
        val_mask = split_column == 'val'
        test_mask = split_column == 'test'
        
        splits = {
            'x_train': x_data[train_mask],
            'x_val': x_data[val_mask],
            'x_test': x_data[test_mask],
            'c_train': c_data[train_mask],
            'c_val': c_data[val_mask],
            'c_test': c_data[test_mask]
        }
        
        print(f"Combination-based split - Train: {np.sum(train_mask)} samples")
        print(f"                        - Val: {np.sum(val_mask)} samples")
        print(f"                        - Test: {np.sum(test_mask)} samples")
        
        # Log parameter ranges for each split
        if 'alpha' in self.df.columns:
            for split_name, mask in [('Train', train_mask), ('Val', val_mask), ('Test', test_mask)]:
                if np.any(mask):
                    alpha_range = [self.df['alpha'][mask].min(), self.df['alpha'][mask].max()]
                    beta_range = [self.df['beta'][mask].min(), self.df['beta'][mask].max()] if 'beta' in self.df.columns else None
                    eta_range = [self.df['eta'][mask].min(), self.df['eta'][mask].max()] if 'eta' in self.df.columns else None
                    print(f"  {split_name} α range: {alpha_range}")
                    if beta_range:
                        print(f"  {split_name} β range: {beta_range}")
                    if eta_range:
                        print(f"  {split_name} η range: {eta_range}")
        
        return splits
    
    
    def _get_split_method_name(self):
        # Check if this is a synthetic dataset
        source_type = self.config.get('dataset', {}).get('source', {}).get('type')
        if source_type in ['synthetic', 'synthetic_vp']:
            return ''  # Store directly in dataset folder without split method subdirectory
        
        preprocessing_config = self.config.get('preprocessing', {})
        split_method = preprocessing_config.get('split_method', 'ratio')
        return split_method
        
    
    def save_tsg_dataset(self, data_splits, task_type=None):
        """
        Save TSG dataset in the format expected by the framework.
        All tasks (cond, int, cf) use the same underlying data.
        """
        output_root = self._setup_output_dirs()
        
        # Create split-specific subdirectory based on split method
        split_method_name = self._get_split_method_name()
        split_dir = f"{output_root}/{split_method_name}"
        os.makedirs(split_dir, exist_ok=True)
        
        dataset_name = self.config['dataset']['name'].lower().replace(' ', '_')
        file_prefix = self.config.get('export', {}).get('file_prefix', dataset_name)
        
        # Save data in TSG format
        file_paths = {}
        
        # Save target and condition data
        for split in ['train', 'val', 'test']:
            # Save with generic names that all tasks can use
            x_path = f"{split_dir}/x_{split}.npy"
            c_path = f"{split_dir}/c_{split}.npy"
            
            np.save(x_path, data_splits[f'x_{split}'])
            np.save(c_path, data_splits[f'c_{split}'])
            
            file_paths[f'x_{split}'] = x_path
            file_paths[f'c_{split}'] = c_path
            
            print(f"Saved {split}: x{data_splits[f'x_{split}'].shape}, c{data_splits[f'c_{split}'].shape}")
        
        # Generate counterfactual data for test split (for synthetic datasets)
        if hasattr(self, 'synthetic_sequences') and self.synthetic_sequences:
            x_cf, c_cf = self._generate_counterfactual_data()
            
            # Save counterfactual data
            x_cf_path = f"{split_dir}/x_cf.npy"
            c_cf_path = f"{split_dir}/c_cf.npy"
            
            np.save(x_cf_path, x_cf)
            np.save(c_cf_path, c_cf)
            
            file_paths['x_cf'] = x_cf_path
            file_paths['c_cf'] = c_cf_path
            
            print(f"Saved counterfactual: x{x_cf.shape}, c{c_cf.shape}")
        
        # Save scalers and encoders
        scalers_path = f"{split_dir}/{file_prefix}_scalers.npz"
        scaler_data = {}
        for name, scaler in self.scalers.items():
            scaler_data[f'{name}_mean'] = scaler.mean_
            scaler_data[f'{name}_scale'] = scaler.scale_
        np.savez(scalers_path, **scaler_data)
        file_paths['scalers'] = scalers_path
        
        # Generate dataset info file
        info_path = self._generate_dataset_info(data_splits, split_dir)
        file_paths['info'] = info_path
        
        print(f"All dataset files saved to: {split_dir}")
        return file_paths
    
    def create_tsg_dataset(self):
        """
        Create a complete TSG dataset that can be used by all tasks.
        """
        print(f"\\n Creating TSG dataset...")
        print("=" * 50)
        
        # Load and preprocess data
        df = self.load_and_preprocess_data()
        
        # Create sequences
        seq_len = self.config.get('preprocessing', {}).get('seq_len', 96)
        interval = self.config.get('preprocessing', {}).get('interval', 1) 
        
        sequences = self.create_sequences(df, seq_len, interval)
        
        # Extract TSG features
        x_data, c_data, _ = self.extract_tsg_features(sequences)
        
        # Normalize data
        x_normalized, c_normalized = self.normalize_data(x_data, c_data)
        
        # Split data
        data_splits = self.split_data(x_normalized, c_normalized)
        
        # Save dataset
        file_paths = self.save_tsg_dataset(data_splits)
        
        print(f"\\nTSG dataset creation completed!")
        return file_paths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create TSG datasets from configuration files')
    parser.add_argument('--config', required=True, help='Path to dataset configuration file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible generation (default: 42)')
    
    args = parser.parse_args()
    np.random.seed(args.seed)

    creator = CaTSGDatasetCreator(args.config)
    files = creator.create_tsg_dataset()
    print(f"Dataset created!! {len(files)} files")
