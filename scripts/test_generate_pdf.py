import asyncio
from pathlib import Path
from datetime import datetime
from quantum_ml_trainer_v6_0 import QuantumMLTrainerV60

async def run_once():
    config = {'exchange': 'asterdex', 'api_key': 'dummy', 'api_secret': 'dummy', 'testnet': True}
    trainer = QuantumMLTrainerV60(config)
    # Use a symbol likely to have enough klines on AsterDEX
    symbol = 'BTCUSDT'
    result = await trainer.train_v60(symbol)
    if result:
        print('Training completed for', symbol)
    await trainer.collector.exchange.close()

if __name__ == '__main__':
    asyncio.run(run_once())
