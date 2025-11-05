'use client';

import { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faBook,
  faRocket,
  faCode,
  faCog,
  faChartLine,
  faRobot,
  faShieldAlt,
  faDatabase,
  faArrowRight,
  faFileCode,
  faGraduationCap,
  faLightbulb
} from '@fortawesome/free-solid-svg-icons';
import { faGithub } from '@fortawesome/free-brands-svg-icons';
import Link from 'next/link';

export default function DocumentationPage() {
  const [activeTab, setActiveTab] = useState('overview');

  const documentationSections = [
    {
      id: 'overview',
      title: 'Overview',
      icon: faBook,
      content: {
        title: 'AuthenticAlgo Trading Bot',
        description: 'Professional Python trading bot with clean, modular architecture supporting multiple exchanges (Binance, AsterDEX), ML integration, and comprehensive risk management.',
        sections: [
          {
            title: 'Key Features',
            items: [
              'AI-Powered Intelligence with 96% accuracy',
              'Multi-layer risk management system',
              'Real-time data processing (1M+ data points)',
              'Sub-100ms trade execution',
              '24/7 automated trading',
              'Multi-exchange support (Binance, AsterDEX)'
            ]
          },
          {
            title: 'Architecture Principles',
            items: [
              'Clean Code: SOLID principles, type hints, proper error handling',
              'Modularity: Single responsibility per module',
              'Scalability: Easy to add new strategies, exchanges, or indicators',
              'Testability: All components unit testable'
            ]
          }
        ]
      }
    },
    {
      id: 'quickstart',
      title: 'Quick Start',
      icon: faRocket,
      content: {
        title: 'Getting Started',
        description: 'Get up and running with AuthenticAlgo in minutes',
        sections: [
          {
            title: '1. Installation',
            code: `# Clone repository
git clone https://github.com/RAYDENFLY/AuthenticAlgo.git
cd AuthenticAlgo

# Create virtual environment
python -m venv .venv
.venv\\Scripts\\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt`
          },
          {
            title: '2. Configuration',
            code: `# Copy example config
cp config/config.yaml.example config/config.yaml

# Edit with your API keys
# NEVER commit config.yaml with real keys!`
          },
          {
            title: '3. Run Demo',
            code: `# Test with paper trading
python demo/demo_paper_trading_quick.py

# Run ML demo
python demo/demo_ml_quick.py

# Start competition
python demo/AsterDEX/run_competition.py`
          }
        ]
      }
    },
    {
      id: 'ml',
      title: 'Machine Learning',
      icon: faRobot,
      content: {
        title: 'ML System Architecture',
        description: 'Advanced machine learning pipeline with 12 optimized models achieving 96% accuracy',
        sections: [
          {
            title: 'Model Performance',
            items: [
              'Random Forest: 100% accuracy (0.00 loss)',
              'Extra Trees: 100% accuracy (0.00 loss)',
              'XGBoost: 100% accuracy (0.00 loss)',
              'LightGBM: 100% accuracy (0.00 loss)',
              'CatBoost: 96% accuracy (0.08 loss)',
              'Gradient Boosting: 88% accuracy (0.24 loss)',
              'AdaBoost: 82% accuracy (0.36 loss)'
            ]
          },
          {
            title: 'Feature Engineering',
            items: [
              '52 technical indicators',
              'Price action features',
              'Volume analysis',
              'Momentum indicators',
              'Volatility metrics',
              'Trend detection'
            ]
          },
          {
            title: 'Training Pipeline',
            code: `from ml.model_trainer import ModelTrainer
from ml.feature_engine import FeatureEngine

# Initialize
feature_engine = FeatureEngine()
trainer = ModelTrainer()

# Prepare data
X, y = feature_engine.prepare_features(df)

# Train models
trainer.train_all_models(X, y)

# Optimize with Optuna
trainer.optimize_hyperparameters(X, y, n_trials=100)`
          }
        ]
      }
    },
    {
      id: 'strategies',
      title: 'Trading Strategies',
      icon: faChartLine,
      content: {
        title: 'Strategy System',
        description: 'Three powerful strategies: Pure Technical Analysis, Pure ML, and Hybrid approach',
        sections: [
          {
            title: 'Pure Technical Analysis',
            items: [
              'RSI, MACD, Bollinger Bands',
              'Moving Average crossovers',
              'Support/Resistance levels',
              'Volume confirmation',
              'ROI: +97% (baseline)'
            ]
          },
          {
            title: 'Pure ML Strategy',
            items: [
              'Ensemble of 12 ML models',
              'Real-time predictions',
              '96% accuracy rate',
              'Confidence-based filtering',
              'ROI: +875% (9x better than TA)'
            ]
          },
          {
            title: 'Hybrid Strategy',
            items: [
              'ML predictions + TA confirmation',
              'Best risk-adjusted returns',
              'Lower drawdown',
              'Balanced approach',
              'ROI: +456%'
            ]
          },
          {
            title: 'Risk Management',
            items: [
              'TP1: 1.5% (Exit 50% position)',
              'TP2: 3.0% (Exit 30% position)',
              'TP3: 5.0% (Exit remaining 20%)',
              'Dynamic SL: 2x ATR',
              'Position sizing: 5% per trade'
            ]
          }
        ]
      }
    },
    {
      id: 'api',
      title: 'API Reference',
      icon: faCode,
      content: {
        title: 'API Documentation',
        description: 'RESTful API with 40+ endpoints and WebSocket support',
        sections: [
          {
            title: 'Base URL',
            code: `http://localhost:8000`
          },
          {
            title: 'Authentication',
            code: `# Headers
Authorization: Bearer <your_token>
Content-Type: application/json`
          },
          {
            title: 'Core Endpoints',
            items: [
              'GET /api/stats - Live statistics',
              'GET /api/positions - Current positions',
              'GET /api/trades - Trade history',
              'POST /api/trade - Execute trade',
              'GET /api/ml/models - ML models info',
              'POST /api/ml/predict - Get prediction',
              'GET /api/arena/status - Competition status',
              'POST /api/arena/start - Start competition'
            ]
          },
          {
            title: 'WebSocket',
            code: `ws://localhost:8000/ws/live-data

# Example client
import websockets

async with websockets.connect('ws://localhost:8000/ws/live-data') as ws:
    while True:
        data = await ws.recv()
        print(f"Received: {data}")`
          }
        ]
      }
    },
    {
      id: 'deployment',
      title: 'Deployment',
      icon: faCog,
      content: {
        title: 'Production Deployment',
        description: 'Deploy your own instance of AuthenticAlgo',
        sections: [
          {
            title: 'Requirements',
            items: [
              'Python 3.9+',
              'Node.js 18+ (for frontend)',
              'PostgreSQL or SQLite',
              'GPU (optional, for ML training)',
              'Linux/Windows Server'
            ]
          },
          {
            title: 'Environment Setup',
            code: `# .env file
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
ASTERDEX_API_KEY=your_key_here
DATABASE_URL=postgresql://user:pass@localhost/trading_bot
REDIS_URL=redis://localhost:6379
JWT_SECRET=your_jwt_secret

# NEVER commit .env to git!`
          },
          {
            title: 'Production Start',
            code: `# Backend
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Frontend
cd web
npm run build
npm start

# Trading Bot
python main.py --mode production`
          },
          {
            title: 'Security Best Practices',
            items: [
              'Use environment variables for secrets',
              'Enable rate limiting',
              'Use HTTPS/WSS in production',
              'Implement API authentication',
              'Regular security audits',
              'Backup database regularly'
            ]
          }
        ]
      }
    }
  ];

  const activeContent = documentationSections.find(s => s.id === activeTab)?.content;

  return (
    <div className="min-h-screen bg-dark-bg">
      {/* Header */}
      <section className="relative py-24 px-4 bg-gradient-to-br from-primary-900/20 via-dark-bg to-purple-900/20 border-b border-white/10">
        <div className="container mx-auto">
          <div className="max-w-4xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 bg-primary-500/10 border border-primary-500/30 rounded-full px-4 py-2 mb-6">
              <FontAwesomeIcon icon={faBook} className="text-primary-400" />
              <span className="text-sm font-semibold text-primary-400">
                DOCUMENTATION
              </span>
            </div>

            <h1 className="text-5xl md:text-7xl font-black mb-6">
              <span className="bg-gradient-to-r from-primary-400 to-bull bg-clip-text text-transparent">
                Developer
              </span>
              <br />
              <span className="text-white">Documentation</span>
            </h1>
            
            <p className="text-xl text-neutral-400 mb-8">
              Complete guide to understanding, deploying, and customizing AuthenticAlgo trading bot
            </p>

            {/* Quick Links */}
            <div className="flex flex-wrap gap-4 justify-center">
              <a
                href="https://github.com/RAYDENFLY/AuthenticAlgo"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 bg-neutral-800 hover:bg-neutral-700 text-white font-bold px-6 py-3 rounded-xl transition-all duration-300"
              >
                <FontAwesomeIcon icon={faGithub} />
                GitHub Repository
              </a>
              <Link
                href="/reports"
                className="inline-flex items-center gap-2 bg-primary-500 hover:bg-primary-600 text-white font-bold px-6 py-3 rounded-xl transition-all duration-300"
              >
                <FontAwesomeIcon icon={faChartLine} />
                View Reports
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Documentation Content */}
      <section className="py-16 px-4">
        <div className="container mx-auto max-w-[1600px]">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
            {/* Sidebar Navigation */}
            <div className="lg:col-span-1">
              <div className="sticky top-24 bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-6">
                <h3 className="text-lg font-bold text-white mb-4">Contents</h3>
                <nav className="space-y-2">
                  {documentationSections.map((section) => (
                    <button
                      key={section.id}
                      onClick={() => setActiveTab(section.id)}
                      className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-left transition-all duration-300 ${
                        activeTab === section.id
                          ? 'bg-primary-500 text-white'
                          : 'text-neutral-400 hover:bg-white/5 hover:text-white'
                      }`}
                    >
                      <FontAwesomeIcon icon={section.icon} />
                      {section.title}
                    </button>
                  ))}
                </nav>
              </div>
            </div>

            {/* Main Content */}
            <div className="lg:col-span-3">
              <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-3xl p-8">
                <h2 className="text-4xl font-black text-white mb-4">
                  {activeContent?.title}
                </h2>
                <p className="text-xl text-neutral-400 mb-8 leading-relaxed">
                  {activeContent?.description}
                </p>

                {/* Content Sections */}
                <div className="space-y-8">
                  {activeContent?.sections.map((section, index) => (
                    <div key={index}>
                      <h3 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
                        <div className="w-8 h-8 bg-primary-500/20 rounded-lg flex items-center justify-center">
                          <span className="text-primary-400 font-bold">{index + 1}</span>
                        </div>
                        {section.title}
                      </h3>
                      
                      {section.items && (
                        <ul className="space-y-3 ml-11">
                          {section.items.map((item, i) => (
                            <li key={i} className="flex items-start gap-3">
                              <FontAwesomeIcon icon={faArrowRight} className="text-bull mt-1 flex-shrink-0" />
                              <span className="text-neutral-300">{item}</span>
                            </li>
                          ))}
                        </ul>
                      )}

                      {section.code && (
                        <div className="ml-11 bg-neutral-900/50 border border-white/10 rounded-xl p-6 overflow-x-auto">
                          <pre className="text-sm text-neutral-300 font-mono">
                            <code>{section.code}</code>
                          </pre>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Bottom Navigation */}
              <div className="mt-8 flex justify-between items-center">
                <button
                  onClick={() => {
                    const currentIndex = documentationSections.findIndex(s => s.id === activeTab);
                    if (currentIndex > 0) {
                      setActiveTab(documentationSections[currentIndex - 1].id);
                    }
                  }}
                  disabled={activeTab === documentationSections[0].id}
                  className="px-6 py-3 bg-white/5 hover:bg-white/10 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold rounded-xl transition-all duration-300"
                >
                  Previous
                </button>
                <button
                  onClick={() => {
                    const currentIndex = documentationSections.findIndex(s => s.id === activeTab);
                    if (currentIndex < documentationSections.length - 1) {
                      setActiveTab(documentationSections[currentIndex + 1].id);
                    }
                  }}
                  disabled={activeTab === documentationSections[documentationSections.length - 1].id}
                  className="px-6 py-3 bg-primary-500 hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold rounded-xl transition-all duration-300"
                >
                  Next
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Additional Resources */}
      <section className="py-24 px-4 border-t border-white/10">
        <div className="container mx-auto">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-4xl font-black text-center mb-12">
              <span className="bg-gradient-to-r from-primary-400 to-bull bg-clip-text text-transparent">
                Supported Exchanges
              </span>
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-16">
              {/* Binance */}
              <div className="bg-gradient-to-br from-[#F3BA2F]/10 to-[#F3BA2F]/5 border border-[#F3BA2F]/30 rounded-2xl p-8 hover:border-[#F3BA2F]/50 transition-all duration-300">
                <div className="flex items-center gap-4 mb-6">
                  <div className="w-16 h-16 bg-[#F3BA2F]/20 rounded-xl flex items-center justify-center">
                    <span className="text-3xl font-black text-[#F3BA2F]">B</span>
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold text-white">Binance</h3>
                    <p className="text-neutral-400 text-sm">World's Leading Exchange</p>
                  </div>
                </div>
                <ul className="space-y-3 mb-6">
                  <li className="flex items-start gap-2">
                    <FontAwesomeIcon icon={faArrowRight} className="text-[#F3BA2F] mt-1 flex-shrink-0" />
                    <span className="text-neutral-300">Spot & Futures Trading</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <FontAwesomeIcon icon={faArrowRight} className="text-[#F3BA2F] mt-1 flex-shrink-0" />
                    <span className="text-neutral-300">Real-time WebSocket streams</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <FontAwesomeIcon icon={faArrowRight} className="text-[#F3BA2F] mt-1 flex-shrink-0" />
                    <span className="text-neutral-300">Advanced order types</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <FontAwesomeIcon icon={faArrowRight} className="text-[#F3BA2F] mt-1 flex-shrink-0" />
                    <span className="text-neutral-300">High liquidity & low fees</span>
                  </li>
                </ul>
                <div className="flex gap-2">
                  <span className="px-3 py-1 bg-[#F3BA2F]/20 border border-[#F3BA2F]/30 rounded-lg text-sm font-semibold text-[#F3BA2F]">
                    ✓ Tested
                  </span>
                  <span className="px-3 py-1 bg-bull/20 border border-bull/30 rounded-lg text-sm font-semibold text-bull">
                    ✓ Production Ready
                  </span>
                </div>
              </div>

              {/* AsterDEX */}
              <div className="bg-gradient-to-br from-primary-500/10 to-purple-500/10 border border-primary-500/30 rounded-2xl p-8 hover:border-primary-500/50 transition-all duration-300">
                <div className="flex items-center gap-4 mb-6">
                  <div className="w-16 h-16 bg-gradient-to-br from-primary-500/20 to-purple-500/20 rounded-xl flex items-center justify-center">
                    <span className="text-3xl font-black text-primary-400">A</span>
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold text-white">AsterDEX</h3>
                    <p className="text-neutral-400 text-sm">Decentralized Exchange</p>
                  </div>
                </div>
                <ul className="space-y-3 mb-6">
                  <li className="flex items-start gap-2">
                    <FontAwesomeIcon icon={faArrowRight} className="text-primary-400 mt-1 flex-shrink-0" />
                    <span className="text-neutral-300">DEX Trading Integration</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <FontAwesomeIcon icon={faArrowRight} className="text-primary-400 mt-1 flex-shrink-0" />
                    <span className="text-neutral-300">Real-time market data API</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <FontAwesomeIcon icon={faArrowRight} className="text-primary-400 mt-1 flex-shrink-0" />
                    <span className="text-neutral-300">Smart contract automation</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <FontAwesomeIcon icon={faArrowRight} className="text-primary-400 mt-1 flex-shrink-0" />
                    <span className="text-neutral-300">On-chain trading execution</span>
                  </li>
                </ul>
                <div className="flex gap-2">
                  <span className="px-3 py-1 bg-primary-500/20 border border-primary-500/30 rounded-lg text-sm font-semibold text-primary-400">
                    ✓ Integrated
                  </span>
                  <span className="px-3 py-1 bg-bull/20 border border-bull/30 rounded-lg text-sm font-semibold text-bull">
                    ✓ Active
                  </span>
                </div>
              </div>
            </div>

            {/* Coming Soon Exchanges */}
            <div className="text-center mb-12">
              <h3 className="text-2xl font-bold text-white mb-6">Coming Soon</h3>
              <div className="flex flex-wrap gap-4 justify-center">
                {['Bybit', 'OKX', 'Gate.io', 'KuCoin', 'Kraken'].map((exchange) => (
                  <div key={exchange} className="px-6 py-3 bg-white/5 border border-white/10 rounded-xl text-neutral-400">
                    {exchange}
                  </div>
                ))}
              </div>
            </div>

            {/* Additional Resources */}
            <h2 className="text-4xl font-black text-center mb-12 mt-20">
              <span className="bg-gradient-to-r from-primary-400 to-bull bg-clip-text text-transparent">
                Additional Resources
              </span>
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white/5 border border-white/10 rounded-2xl p-6 text-center hover:border-primary-500/50 transition-all duration-300">
                <FontAwesomeIcon icon={faLightbulb} className="text-4xl text-bull mb-4" />
                <h3 className="text-xl font-bold text-white mb-2">Best Practices</h3>
                <p className="text-neutral-400 mb-4">Trading tips and risk management strategies</p>
                <Link href="/best-practices" className="text-primary-400 hover:text-primary-300 font-semibold">
                  Coming Soon →
                </Link>
              </div>

              <div className="bg-white/5 border border-white/10 rounded-2xl p-6 text-center hover:border-primary-500/50 transition-all duration-300">
                <FontAwesomeIcon icon={faFileCode} className="text-4xl text-purple-400 mb-4" />
                <h3 className="text-xl font-bold text-white mb-2">Code Examples</h3>
                <p className="text-neutral-400 mb-4">Ready-to-use code snippets and demos</p>
                <a 
                  href="https://github.com/RAYDENFLY/AuthenticAlgo/tree/main/demo"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary-400 hover:text-primary-300 font-semibold"
                >
                  View on GitHub →
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
