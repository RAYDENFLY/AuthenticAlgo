'use client';

import { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faChartLine, 
  faRobot, 
  faDownload,
  faCalendar,
  faFileAlt,
  faTrophy,
  faArrowUp,
  faArrowDown
} from '@fortawesome/free-solid-svg-icons';
import Link from 'next/link';

export default function ReportsPage() {
  const [selectedPeriod, setSelectedPeriod] = useState('all');

  const reports = [
    {
      id: 1,
      title: 'Complete Strategy Comparison Report',
      description: 'Comprehensive analysis comparing Pure Technical Analysis, Pure ML, and Hybrid strategies over 10,000 trades',
      date: 'November 2025',
      file: 'COMPLETE_STRATEGY_COMPARISON.md',
      category: 'Strategy Analysis',
      highlights: [
        'Pure ML: 9x better ROI than TA',
        'Hybrid: Best risk-adjusted returns',
        '10,000 trades analyzed'
      ],
      metrics: {
        trades: 10000,
        accuracy: '96%',
        roi: '+875%'
      }
    },
    {
      id: 2,
      title: 'Machine Learning Optimization Report',
      description: 'ML model optimization using Optuna achieving 75-100% accuracy across 11/12 models',
      date: 'November 2025',
      file: 'ML_OPTIMIZATION_SUCCESS.md',
      category: 'ML Performance',
      highlights: [
        '11/12 models optimized',
        'Accuracy: 75-100%',
        'Optuna hyperparameter tuning'
      ],
      metrics: {
        models: 12,
        accuracy: '96%',
        improvement: '+120%'
      }
    },
    {
      id: 3,
      title: 'ML Validation Results',
      description: 'Validation on unseen data showing 85-96% test accuracy across production models',
      date: 'November 2025',
      file: 'ML_VALIDATION_RESULTS.md',
      category: 'ML Performance',
      highlights: [
        '4/5 models passed validation',
        'Test accuracy: 85-96%',
        'Production-ready models'
      ],
      metrics: {
        testAccuracy: '91%',
        precision: '89%',
        recall: '93%'
      }
    },
    {
      id: 4,
      title: 'Final Strategy Comparison',
      description: 'Head-to-head comparison of all trading strategies with detailed performance metrics',
      date: 'November 2025',
      file: 'FINAL_COMPARISON_REPORT.md',
      category: 'Strategy Analysis',
      highlights: [
        'Pure ML leads with 9x ROI',
        'Detailed risk analysis',
        'Trade-by-trade breakdown'
      ],
      metrics: {
        strategies: 3,
        totalTrades: 30,
        bestROI: '+875%'
      }
    },
    {
      id: 5,
      title: 'ML Benchmark Report',
      description: 'Comprehensive benchmark of 12 machine learning models on trading data',
      date: 'November 2025',
      file: 'ML_BENCHMARK_REPORT.md',
      category: 'ML Performance',
      highlights: [
        '12 models tested',
        'Feature importance analysis',
        'Model comparison matrix'
      ],
      metrics: {
        models: 12,
        features: 52,
        accuracy: '96%'
      }
    },
    {
      id: 6,
      title: 'AsterDEX Integration Guide',
      description: 'Complete guide for integrating with AsterDEX exchange API',
      date: 'November 2025',
      file: 'ASTERDEX_INTEGRATION.md',
      category: 'Integration',
      highlights: [
        'Real-time data streaming',
        'Order execution',
        'Risk management'
      ],
      metrics: {
        latency: '<100ms',
        uptime: '99.9%',
        pairs: '50+'
      }
    },
    {
      id: 7,
      title: 'Data Collection Report',
      description: 'Analysis of data collection infrastructure and market data quality',
      date: 'November 2025',
      file: 'DATA_COLLECTION_REPORT.md',
      category: 'Infrastructure',
      highlights: [
        '1M+ data points/day',
        'Real-time processing',
        'Multi-exchange support'
      ],
      metrics: {
        dataPoints: '1M+',
        exchanges: 3,
        uptime: '99.9%'
      }
    }
  ];

  const categories = ['All', 'Strategy Analysis', 'ML Performance', 'Integration', 'Infrastructure'];

  return (
    <div className="min-h-screen bg-dark-bg">
      {/* Header */}
      <section className="relative py-24 px-4 bg-gradient-to-br from-primary-900/20 via-dark-bg to-purple-900/20 border-b border-white/10">
        <div className="container mx-auto">
          <div className="max-w-4xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 bg-primary-500/10 border border-primary-500/30 rounded-full px-4 py-2 mb-6">
              <FontAwesomeIcon icon={faChartLine} className="text-primary-400" />
              <span className="text-sm font-semibold text-primary-400">
                PERFORMANCE REPORTS
              </span>
            </div>

            <h1 className="text-5xl md:text-7xl font-black mb-6">
              <span className="bg-gradient-to-r from-primary-400 to-bull bg-clip-text text-transparent">
                Trading Bot
              </span>
              <br />
              <span className="text-white">Performance Reports</span>
            </h1>
            
            <p className="text-xl text-neutral-400 mb-8">
              Comprehensive analysis of our AI-powered trading strategies, ML model performance, and system infrastructure
            </p>

            {/* Quick Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-3xl mx-auto">
              {[
                { value: '7', label: 'Reports', icon: faFileAlt },
                { value: '96%', label: 'ML Accuracy', icon: faRobot },
                { value: '+875%', label: 'Best ROI', icon: faTrophy },
                { value: '10K+', label: 'Trades Analyzed', icon: faChartLine }
              ].map((stat, index) => (
                <div key={index} className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-4">
                  <FontAwesomeIcon icon={stat.icon} className="text-2xl text-primary-400 mb-2" />
                  <div className="text-2xl font-bold text-white mb-1">{stat.value}</div>
                  <div className="text-sm text-neutral-500">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Category Filter */}
      <section className="py-8 px-4 border-b border-white/10">
        <div className="container mx-auto">
          <div className="flex flex-wrap gap-3 justify-center">
            {categories.map((category) => (
              <button
                key={category}
                onClick={() => setSelectedPeriod(category.toLowerCase())}
                className={`px-6 py-2 rounded-xl font-semibold transition-all duration-300 ${
                  selectedPeriod === category.toLowerCase()
                    ? 'bg-primary-500 text-white'
                    : 'bg-white/5 text-neutral-400 hover:bg-white/10 hover:text-white'
                }`}
              >
                {category}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Reports Grid */}
      <section className="py-16 px-4">
        <div className="container mx-auto max-w-[1600px]">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {reports.map((report) => (
              <div
                key={report.id}
                className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-3xl p-8 hover:border-primary-500/50 transition-all duration-300 group"
              >
                {/* Header */}
                <div className="flex items-start justify-between mb-6">
                  <div>
                    <span className="inline-block px-3 py-1 bg-primary-500/20 border border-primary-500/30 rounded-lg text-sm font-semibold text-primary-400 mb-3">
                      {report.category}
                    </span>
                    <h3 className="text-2xl font-bold text-white mb-2 group-hover:text-primary-400 transition-colors">
                      {report.title}
                    </h3>
                    <div className="flex items-center gap-2 text-neutral-500 text-sm">
                      <FontAwesomeIcon icon={faCalendar} />
                      {report.date}
                    </div>
                  </div>
                </div>

                {/* Description */}
                <p className="text-neutral-400 mb-6 leading-relaxed">
                  {report.description}
                </p>

                {/* Highlights */}
                <div className="mb-6">
                  <h4 className="text-sm font-semibold text-neutral-500 mb-3">KEY HIGHLIGHTS</h4>
                  <div className="space-y-2">
                    {report.highlights.map((highlight, index) => (
                      <div key={index} className="flex items-start gap-2">
                        <div className="w-1.5 h-1.5 bg-bull rounded-full mt-2 flex-shrink-0"></div>
                        <span className="text-neutral-300 text-sm">{highlight}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-3 gap-4 mb-6 p-4 bg-white/5 rounded-xl border border-white/10">
                  {Object.entries(report.metrics).map(([key, value]) => (
                    <div key={key} className="text-center">
                      <div className="text-lg font-bold text-bull">{value}</div>
                      <div className="text-xs text-neutral-500 capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</div>
                    </div>
                  ))}
                </div>

                {/* Actions */}
                <div className="flex gap-3">
                  <Link
                    href={`/${report.file}`}
                    className="flex-1 bg-primary-500 hover:bg-primary-600 text-white font-bold py-3 px-6 rounded-xl transition-all duration-300 text-center"
                  >
                    <FontAwesomeIcon icon={faFileAlt} className="mr-2" />
                    View Report
                  </Link>
                  <a
                    href={`/${report.file}`}
                    download
                    className="border-2 border-white/20 hover:border-primary-400 text-white hover:text-primary-400 font-bold py-3 px-6 rounded-xl transition-all duration-300"
                  >
                    <FontAwesomeIcon icon={faDownload} />
                  </a>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-4 border-t border-white/10">
        <div className="container mx-auto text-center">
          <div className="max-w-3xl mx-auto">
            <h2 className="text-4xl md:text-5xl font-black mb-6">
              <span className="bg-gradient-to-r from-primary-400 to-bull bg-clip-text text-transparent">
                Want More Details?
              </span>
            </h2>
            <p className="text-xl text-neutral-400 mb-8">
              Join our Discord community to access exclusive reports, live trading data, and discuss strategies with other traders
            </p>
            <Link
              href="/"
              className="inline-flex items-center gap-3 bg-primary-500 hover:bg-primary-600 text-white font-bold text-lg px-10 py-4 rounded-2xl transition-all duration-300 transform hover:scale-105"
            >
              <FontAwesomeIcon icon={faChartLine} />
              Back to Home
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
