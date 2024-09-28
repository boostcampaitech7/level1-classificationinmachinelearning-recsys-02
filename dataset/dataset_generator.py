import os
from typing import List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm

class DatasetGenerator:
    COLS_DICT: Dict[str, str] = {
        "ID": "ID",
        "target": "target",
        "_type": "_type",
        # coinbase
        "hourly_market-data_coinbase-premium-index_coinbase_premium_gap": "coinbase_premium_gap",
        "hourly_market-data_coinbase-premium-index_coinbase_premium_index": "coinbase_premium_index",
        # funding_rate
        "hourly_market-data_funding-rates_all_exchange_funding_rates": "funding_rates",
        "hourly_market-data_funding-rates_bitmex_funding_rates": "funding_rates",
        
        "hourly_market-data_funding-rates_bitmex_funding_rates": "funding_rates_bitmex",
        # liquidations
        "hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations": "long_liquidations",
        "hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations_usd": "long_liquidations_usd",
        "hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations": "short_liquidations",
        "hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations_usd": "short_liquidations_usd",
        
        "hourly_market-data_liquidations_bybit_all_symbol_long_liquidations": "long_liquidations_bybit",
        "hourly_market-data_liquidations_bybit_all_symbol_short_liquidations": "short_liquidations_bybit",
        "hourly_market-data_liquidations_bitfinex_all_symbol_long_liquidations": "long_liquidations_bitfinex",
        "hourly_market-data_liquidations_bitfinex_all_symbol_short_liquidations": "short_liquidations_bitfinex",
        "hourly_market-data_liquidations_binance_all_symbol_long_liquidations": "long_liquidations_binance",
        "hourly_market-data_liquidations_binance_all_symbol_short_liquidations": "short_liquidations_binance",
        
        # open-interest
        "hourly_market-data_open-interest_all_exchange_all_symbol_open_interest": "open_interest",
        
        # taker_buy_sell
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_ratio": "buy_ratio",
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_sell_ratio": "buy_sell_ratio",
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume": "buy_volume",
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_ratio": "sell_ratio",
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume": "sell_volume",
        
        'hourly_market-data_taker-buy-sell-stats_huobi_global_taker_buy_sell_ratio' : "buy_sell_ratio_huobi",
        "hourly_market-data_taker-buy-sell-stats_deribit_taker_buy_volume": "buy_volume_deribit",
        "hourly_market-data_taker-buy-sell-stats_deribit_taker_sell_volume": "sell_volume_deribit",
        "hourly_market-data_taker-buy-sell-stats_bybit_taker_buy_volume": "buy_volume_bybit",
        "hourly_market-data_taker-buy-sell-stats_bybit_taker_sell_volume": "sell_volume_bybit",
        "hourly_market-data_taker-buy-sell-stats_okx_taker_buy_volume": "buy_volume_okx",
        "hourly_market-data_taker-buy-sell-stats_okx_taker_sell_volume": "sell_volume_okx",
        
        # address
        "hourly_network-data_addresses-count_addresses_count_active": "active_count",
        "hourly_network-data_addresses-count_addresses_count_receiver": "receiver_count",
        "hourly_network-data_addresses-count_addresses_count_sender": "sender_count",
        
        # transactions_count
        'hourly_network-data_transactions-count_transactions_count_total' : "transactions_count_total",
        'hourly_network-data_transactions-count_transactions_count_mean' : "transactions_count_mean",
        
        # fees_block
        'hourly_network-data_fees_fees_block_mean' : "block_mean",
        'hourly_network-data_fees_fees_block_mean_usd' : "block_mean_usd",
        
        # fees
        'hourly_network-data_fees_fees_total' : 'fees_total',
        'hourly_network-data_fees_fees_total_usd' : 'fees_total_usd',
        'hourly_network-data_fees_fees_reward_percent' : 'fees_reward_percent',
        
        # difficulty
        'hourly_network-data_difficulty_difficulty' : 'difficulty',
        
        # utxo
        'hourly_network-data_utxo-count_utxo_count' : 'utxo_count',
        
        # supply
        'hourly_network-data_supply_supply_total' : 'supply_total',
        'hourly_network-data_supply_supply_new' : 'supply_new',
        
        # hashrate
        'hourly_network-data_hashrate_hashrate' : 'hashrate',
        
        # fees_transaction
        'hourly_network-data_fees-transaction_fees_transaction_mean' : 'fees_transaction_mean',
        'hourly_network-data_fees-transaction_fees_transaction_mean_usd' : 'fees_transaction_mean_usd',
        'hourly_network-data_fees-transaction_fees_transaction_median' : 'fees_transaction_median',
        'hourly_network-data_fees-transaction_fees_transaction_median_usd' : 'fees_transaction_median_usd',
        
        # blockreward
        'hourly_network-data_blockreward_blockreward' : 'blockreward',
        'hourly_network-data_blockreward_blockreward_usd' : 'blockreward_usd',
        'hourly_network-data_block-interval_block_interval' : 'block_interval',
        
        # transffered_tokens
        'hourly_network-data_tokens-transferred_tokens_transferred_total' : 'tokens_transferred_total',
        'hourly_network-data_tokens-transferred_tokens_transferred_mean' : 'tokens_transferred_mean',
        'hourly_network-data_tokens-transferred_tokens_transferred_median' : 'tokens_transferred_median',
        
        # block_bytes
        'hourly_network-data_block-bytes_block_bytes' : 'block_bytes',
        
        # velocity
        'hourly_network-data_velocity_velocity_supply_total' : 'velocity_supply_total'
    }

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df: pd.DataFrame = None
        self.category_cols: List[str] = []
        self.conti_cols: List[str] = []

    def load_data(self) -> None:
        """Load and merge train and test data"""
        train_df = pd.read_csv(os.path.join(self.data_path, "train.csv")).assign(_type="train")
        test_df = pd.read_csv(os.path.join(self.data_path, "test.csv")).assign(_type="test")
        self.df = pd.concat([train_df, test_df], axis=0)

    def load_hourly_data(self) -> None:
        """Load and merge hourly data"""
        # HOURLY_ 로 시작하는 .csv 파일 이름을 file_names 에 할딩
        file_names: List[str] = [
            f for f in os.listdir(self.data_path) if f.startswith("HOURLY_") and f.endswith(".csv")
        ]

        # 파일명 : 데이터프레임으로 딕셔너리 형태로 저장
        file_dict: Dict[str, pd.DataFrame] = {
            f.replace(".csv", ""): pd.read_csv(os.path.join(self.data_path, f)) for f in file_names
        }

        for _file_name, _df in tqdm(file_dict.items()):
            # 열 이름 중복 방지를 위해 {_file_name.lower()}_{col.lower()}로 변경, datetime 열을 ID로 변경
            _rename_rule = {
                col: f"{_file_name.lower()}_{col.lower()}" if col != "datetime" else "ID"
                for col in _df.columns
            }
            _df = _df.rename(_rename_rule, axis=1)
            self.df = self.df.merge(_df, on="ID", how="left")

    def make_df(self) -> pd.DataFrame:
        """Select and rename columns"""
        return self.df[self.COLS_DICT.keys()].rename(columns=self.COLS_DICT)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features and preprocess data"""
        epsilon = 1e-6

        conti_features = {
            "liquidation_diff": df["long_liquidations"] - df["short_liquidations"],
            "liquidation_diff_bybit": df["long_liquidations_bybit"] - df["short_liquidations_bybit"],
            "liquidation_index": (df["long_liquidations"] - df["short_liquidations"]) / (df["long_liquidations"] + df["short_liquidations"] + epsilon),
            "liquidation_index_bitfinex": (df["long_liquidations_bitfinex"] - df["short_liquidations_bitfinex"]) / (df["long_liquidations_bitfinex"] + df["short_liquidations_bitfinex"] + epsilon),
            "liquidation_index_binance": (df["long_liquidations_binance"] - df["short_liquidations_binance"]) / (df["long_liquidations_binance"] + df["short_liquidations_binance"] + epsilon),
            
            "volume_diff": df["buy_volume"] - df["sell_volume"],
            "volume_diff_deribit": df["buy_volume_deribit"] - df["sell_volume_deribit"],
            "volume_diff_bybit": df["buy_volume_bybit"] - df["sell_volume_bybit"],
            "volume_diff_okx": df["buy_volume_okx"] - df["sell_volume_okx"],
            "volume_index": (df["buy_volume"] - df["sell_volume"]) / (df["buy_volume"] + df["sell_volume"] + epsilon),
            
            "address_diff": df["sender_count"] - df["receiver_count"],
            
            "buy_sell_volume_ratio" : df["buy_volume"] / (df["sell_volume"] + epsilon),
            "long_liquidation_interest_ratio" : df["long_liquidations"]/df['open_interest'],
            "short_liquidation_interest_ratio" : df["short_liquidations"]/df['open_interest'],
            "long_liquidation_volume_ratio" : df["long_liquidations"]/(df['buy_volume']+df['sell_volume']+ epsilon),
            "short_liquidation_volume_ratio" : df["short_liquidations"]/(df['buy_volume']+df['sell_volume']+ epsilon),
            "volume_interest_ratio" : (df['buy_volume']+df['sell_volume'])/(df['open_interest']+ epsilon),

            "market_pressure" : (df['buy_ratio']*df['open_interest']) / (df['sell_ratio']*df['funding_rates']+epsilon),
            "network_active" : (df['active_count']*df['transactions_count_total'])/(df['block_interval']*df['block_bytes']+epsilon),
            "Hodler" : (df['utxo_count']*df['difficulty'])/(df['velocity_supply_total']*df['supply_new']+epsilon),
            "profitability" : (df['blockreward']*df['hashrate'])/(df['difficulty']*df['fees_total']+epsilon),
            "investment" : df['coinbase_premium_index']*df['tokens_transferred_mean']/(df['tokens_transferred_median']+epsilon),
            "leverage" : (df['long_liquidations'] + df['short_liquidations'])*df['open_interest']/(df['active_count']*df['tokens_transferred_total']+epsilon),
            "fee_index" : (df['fees_transaction_mean'] * df['transactions_count_total'])/(df['active_count']*df['supply_new']+epsilon),
            "market_health" : (df['hashrate']*df['active_count'])/(df['long_liquidations']+df['short_liquidations']+epsilon),
            "exchange_center" : (df['tokens_transferred_total'] - df['tokens_transferred_mean']*df['transactions_count_total'])/(df['utxo_count']+epsilon),
            
            "hashrate_difficulty_reward_ratio" : (df['hashrate'] * df['blockreward']) / (df['difficulty'] * df['fees_total']),
            "fees_activity_ratio" : df['fees_total'] / (df['active_count'] * df['transactions_count_total']), 
            "tokens_fee_ratio" : df['tokens_transferred_mean'] / df['fees_transaction_mean'],
            "block_interval_difficulty_reward_ratio" : (df['blockreward'] / df['block_interval']) * (df['difficulty'] / df['fees_reward_percent']), 
            "taker_interest_ratio" : (df['buy_volume'] + df['sell_volume']) / df['open_interest'] ,
            "premium_liquidation_ratio" : df['fees_total'] / (df['difficulty'] * df['utxo_count']),
            "velocity_supply_ratio" : df['velocity_supply_total'] / (df['supply_total'] * df['transactions_count_mean']),
            "funding_taker_ratio" : df['buy_sell_ratio'] / df['funding_rates'],
        }
        
        category_features = {
            # 추가
            "M" : pd.to_datetime(df['ID']).dt.month,
            'Is_Afternoon_Evening' : pd.to_datetime(df['ID']).dt.hour.apply(lambda x: 1 if 12 <= x <= 20 else 0),
            'Is_Weekend' : pd.to_datetime(df['ID']).dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
        }

        df = df.assign(**conti_features)
        df = df.assign(**category_features)

        self.category_cols = list(category_features.keys())
        self.conti_cols = [col for col in self.COLS_DICT.values() if col not in ["ID", "target", "_type"]] + list(conti_features.keys())

        return df, self.category_cols, self.conti_cols
    
    def moving_average(df, conti_cols, intervals):
        """Create moving average feature"""
        df_ma_dict = [
            df[conti_col].rolling(window=interval).mean().rename(f"{conti_col}_MA{interval}")
            for conti_col in conti_cols
            for interval in intervals
        ]
    
        df_ma = pd.concat([df, pd.concat(df_ma_dict, axis=1)], axis=1)
        return df_ma
    
    def shift_feature(df, conti_cols, intervals) -> List[pd.Series]:
        """Create shift feature"""
        df_shift_dict = [
            df[conti_col].shift(interval).rename(f"{conti_col}_{interval}")
            for conti_col in conti_cols
            for interval in intervals
        ]
        
        df_shift = pd.concat([df, pd.concat(df_shift_dict, axis=1)], axis=1)
        return df_shift

    def generate_dataset(self) -> pd.DataFrame:
        """Generate the complete dataset"""
        self.load_data()
        self.load_hourly_data()
        return self.make_df()
    