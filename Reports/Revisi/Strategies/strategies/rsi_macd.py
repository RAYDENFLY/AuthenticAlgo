def __init__(self, config: Dict[str, Any]):
    super().__init__("RSI_MACD_Strategy", config)
    
    # Load optimized parameters atau use defaults
    optimized_params = config.get('optimized_parameters', {})
    self.rsi_period = optimized_params.get('rsi_period', 14)
    self.rsi_oversold = optimized_params.get('rsi_oversold', 30) 
    self.rsi_overbought = optimized_params.get('rsi_overbought', 70)
    self.macd_fast = optimized_params.get('macd_fast', 12)
    self.macd_slow = optimized_params.get('macd_slow', 26)
    
    # Additional optimization dari benchmark results
    self.min_confidence = config.get('min_confidence', 0.6)
    self.require_volume_confirmation = config.get('require_volume_confirmation', True)