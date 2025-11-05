'use client';

import { useEffect, useRef, useState } from 'react';
import anime from 'animejs';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faChartLine,
  faChartBar,
  faChartPie,
  faArrowTrendUp,
  faArrowTrendDown,
  faClock,
  faCalendar,
  faBolt,
  faFire,
  faTrophy,
  faCoins,
  faPercent,
  faExchange,
  faCircle,
  faCheckCircle,
  faTimesCircle,
} from '@fortawesome/free-solid-svg-icons';

export default function AnalyticsPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [timeRange, setTimeRange] = useState('7d');
  
  useEffect(() => {
    if (containerRef.current) {
      anime({
        targets: '.fade-in-card',
        opacity: [0, 1],
        translateY: [30, 0],
        delay: anime.stagger(80),
        duration: 800,
        easing: 'easeOutCubic',
      });
    }
  }, []);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  };

  const getPriceClass = (value: number) => {
    if (value > 0) return 'text-bull';
    if (value < 0) return 'text-bear';
    return 'text-neutral-300';
  };

  // Mock data
  const overviewStats = [
    {
      icon: faCoins,
      label: 'Total Profit',
      value: formatCurrency(45678.92),
      change: 23.4,
      description: 'Last 7 days',
      color: 'from-bull/20 to-bull/5',
      iconColor: 'text-bull',
    },
    {
      icon: faChartLine,
      label: 'Win Rate',
      value: '68.5%',
      change: 5.2,
      description: '156 of 228 trades',
      color: 'from-primary-500/20 to-primary-500/5',
      iconColor: 'text-primary-400',
    },
    {
      icon: faExchange,
      label: 'Total Trades',
      value: '1,247',
      change: 12.8,
      description: 'All strategies',
      color: 'from-purple-500/20 to-purple-500/5',
      iconColor: 'text-purple-400',
    },
    {
      icon: faBolt,
      label: 'Avg Leverage',
      value: '3.2x',
      change: -2.1,
      description: 'Optimal range',
      color: 'from-orange-500/20 to-orange-500/5',
      iconColor: 'text-orange-400',
    },
  ];

  const performanceByStrategy = [
    {
      name: 'ML Strategy',
      profit: 28456.78,
      roi: 142.3,
      trades: 487,
      winRate: 72.1,
      avgDuration: '4.2h',
      color: 'from-purple-500 to-pink-500',
      icon: faChartLine,
    },
    {
      name: 'Technical',
      profit: 12234.56,
      roi: 61.2,
      trades: 398,
      winRate: 65.8,
      avgDuration: '3.8h',
      color: 'from-blue-500 to-cyan-500',
      icon: faChartBar,
    },
    {
      name: 'Hybrid',
      profit: 4987.58,
      roi: 24.9,
      trades: 362,
      winRate: 58.3,
      avgDuration: '5.1h',
      color: 'from-orange-500 to-red-500',
      icon: faChartPie,
    },
  ];

  const monthlyPerformance = [
    { month: 'May', profit: 8234.56, trades: 156, winRate: 64.2 },
    { month: 'Jun', profit: 12456.78, trades: 178, winRate: 67.4 },
    { month: 'Jul', profit: 15678.90, trades: 192, winRate: 69.8 },
    { month: 'Aug', profit: 18234.56, trades: 204, winRate: 71.5 },
    { month: 'Sep', profit: 21456.78, trades: 218, winRate: 68.9 },
    { month: 'Oct', profit: 25678.90, trades: 234, winRate: 70.2 },
  ];

  const topPairs = [
    { pair: 'BTC/USDT', profit: 18234.56, trades: 342, winRate: 71.3, volume: 245678 },
    { pair: 'ETH/USDT', profit: 14567.89, trades: 298, winRate: 68.9, volume: 198234 },
    { pair: 'BNB/USDT', profit: 8456.78, trades: 187, winRate: 65.2, volume: 134567 },
    { pair: 'SOL/USDT', profit: 4234.56, trades: 142, winRate: 62.7, volume: 89234 },
  ];

  const timeDistribution = [
    { time: '00-04', trades: 45, profit: 1234.56, winRate: 62.2 },
    { time: '04-08', trades: 78, profit: 2345.67, winRate: 65.4 },
    { time: '08-12', trades: 156, profit: 5678.90, winRate: 71.2 },
    { time: '12-16', trades: 198, profit: 7890.12, winRate: 69.8 },
    { time: '16-20', trades: 142, profit: 4567.89, winRate: 67.3 },
    { time: '20-24', trades: 98, profit: 3234.56, winRate: 64.5 },
  ];

  return (
    <div className="min-h-screen bg-dark-bg py-24 px-4">
      <div className="mx-auto max-w-[1600px]" ref={containerRef}>
        {/* Header */}
        <div className="mb-12 fade-in-card">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-6">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 bg-gradient-to-br from-primary-500 to-purple-500 rounded-2xl flex items-center justify-center shadow-xl shadow-primary-500/25">
                <FontAwesomeIcon icon={faChartLine} className="text-white text-2xl" />
              </div>
              <div>
                <h1 className="text-5xl font-black text-white mb-2">
                  Analytics
                </h1>
                <p className="text-lg text-neutral-400">
                  Comprehensive trading performance insights
                </p>
              </div>
            </div>

            {/* Time Range Selector */}
            <div className="flex gap-2 bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-2">
              {['24h', '7d', '30d', '90d', 'All'].map((range) => (
                <button
                  key={range}
                  onClick={() => setTimeRange(range)}
                  className={`px-4 py-2 rounded-lg font-bold text-sm transition-all duration-300 ${
                    timeRange === range
                      ? 'bg-primary-500 text-white'
                      : 'text-neutral-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  {range}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Overview Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {overviewStats.map((stat, index) => (
            <div key={index} className="fade-in-card">
              <div className={`bg-gradient-to-br ${stat.color} border border-white/10 rounded-3xl p-6 hover:border-primary-500/50 transition-all duration-300 h-full`}>
                <div className="flex items-center justify-between mb-4">
                  <div className={`w-12 h-12 bg-white/10 rounded-2xl flex items-center justify-center`}>
                    <FontAwesomeIcon icon={stat.icon} className={`${stat.iconColor} text-xl`} />
                  </div>
                  <div className={`px-3 py-1 rounded-xl ${stat.change >= 0 ? 'bg-bull/20 text-bull' : 'bg-bear/20 text-bear'}`}>
                    <FontAwesomeIcon icon={stat.change >= 0 ? faArrowTrendUp : faArrowTrendDown} className="text-xs mr-1" />
                    {Math.abs(stat.change).toFixed(1)}%
                  </div>
                </div>
                <div className="mb-2">
                  <div className="text-neutral-400 text-sm font-semibold mb-2 uppercase tracking-wider">
                    {stat.label}
                  </div>
                  <div className="text-3xl font-black text-white">
                    {stat.value}
                  </div>
                </div>
                <div className="text-sm text-neutral-500 pt-3 border-t border-white/10">
                  {stat.description}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Performance by Strategy */}
        <div className="mb-12">
          <div className="flex items-center gap-4 mb-8 fade-in-card">
            <div className="w-12 h-12 bg-primary-500/20 rounded-2xl flex items-center justify-center">
              <FontAwesomeIcon icon={faTrophy} className="text-primary-400 text-xl" />
            </div>
            <h2 className="text-3xl font-black text-white">Strategy Performance</h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {performanceByStrategy.map((strategy, index) => (
              <div key={index} className="fade-in-card">
                <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-3xl p-6 hover:border-primary-500/50 transition-all duration-300">
                  <div className="flex items-center gap-3 mb-6">
                    <div className={`w-12 h-12 rounded-2xl bg-gradient-to-br ${strategy.color} flex items-center justify-center`}>
                      <FontAwesomeIcon icon={strategy.icon} className="text-white text-xl" />
                    </div>
                    <h3 className="text-xl font-black text-white">{strategy.name}</h3>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <div className="text-neutral-400 text-xs font-bold mb-1 uppercase">Total Profit</div>
                      <div className="text-2xl font-black text-bull">{formatCurrency(strategy.profit)}</div>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div className="bg-white/5 rounded-xl p-3 border border-white/10">
                        <div className="text-neutral-500 text-xs font-bold mb-1 uppercase">ROI</div>
                        <div className={`font-bold ${getPriceClass(strategy.roi)}`}>
                          {formatPercentage(strategy.roi)}
                        </div>
                      </div>
                      <div className="bg-white/5 rounded-xl p-3 border border-white/10">
                        <div className="text-neutral-500 text-xs font-bold mb-1 uppercase">Win Rate</div>
                        <div className="text-white font-bold">{strategy.winRate.toFixed(1)}%</div>
                      </div>
                      <div className="bg-white/5 rounded-xl p-3 border border-white/10">
                        <div className="text-neutral-500 text-xs font-bold mb-1 uppercase">Trades</div>
                        <div className="text-white font-bold">{strategy.trades}</div>
                      </div>
                      <div className="bg-white/5 rounded-xl p-3 border border-white/10">
                        <div className="text-neutral-500 text-xs font-bold mb-1 uppercase">Avg Time</div>
                        <div className="text-white font-bold">{strategy.avgDuration}</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Monthly Performance & Top Pairs */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-12">
          {/* Monthly Performance */}
          <div className="fade-in-card">
            <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-3xl p-8">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 bg-primary-500/20 rounded-2xl flex items-center justify-center">
                  <FontAwesomeIcon icon={faCalendar} className="text-primary-400 text-xl" />
                </div>
                <h3 className="text-2xl font-black text-white">Monthly Trend</h3>
              </div>

              <div className="space-y-4">
                {monthlyPerformance.map((month, index) => {
                  const maxProfit = Math.max(...monthlyPerformance.map(m => m.profit));
                  const widthPercent = (month.profit / maxProfit) * 100;

                  return (
                    <div key={index}>
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-3">
                          <span className="text-neutral-400 font-bold text-sm w-8">{month.month}</span>
                          <span className="text-white font-bold">{formatCurrency(month.profit)}</span>
                        </div>
                        <div className="flex items-center gap-3 text-sm">
                          <span className="text-neutral-500">{month.trades} trades</span>
                          <span className="text-primary-400 font-bold">{month.winRate.toFixed(1)}%</span>
                        </div>
                      </div>
                      <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-primary-500 to-purple-500 rounded-full transition-all duration-1000"
                          style={{ width: `${widthPercent}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Top Trading Pairs */}
          <div className="fade-in-card">
            <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-3xl p-8">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 bg-primary-500/20 rounded-2xl flex items-center justify-center">
                  <FontAwesomeIcon icon={faCoins} className="text-primary-400 text-xl" />
                </div>
                <h3 className="text-2xl font-black text-white">Top Pairs</h3>
              </div>

              <div className="space-y-3">
                {topPairs.map((pair, index) => (
                  <div key={index} className="bg-white/5 rounded-xl p-4 border border-white/10 hover:border-primary-500/50 transition-all">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className={`w-8 h-8 rounded-xl flex items-center justify-center font-bold text-sm ${
                          index === 0 ? 'bg-gradient-to-br from-yellow-500 to-orange-500 text-white' : 'bg-white/10 text-neutral-400'
                        }`}>
                          {index + 1}
                        </div>
                        <div>
                          <div className="text-white font-black">{pair.pair}</div>
                          <div className="text-neutral-500 text-xs">{pair.trades} trades</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-bull font-black">{formatCurrency(pair.profit)}</div>
                        <div className="text-primary-400 text-xs font-bold">{pair.winRate.toFixed(1)}%</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Time Distribution */}
        <div className="fade-in-card">
          <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-3xl p-8">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-12 h-12 bg-primary-500/20 rounded-2xl flex items-center justify-center">
                <FontAwesomeIcon icon={faClock} className="text-primary-400 text-xl" />
              </div>
              <div>
                <h3 className="text-2xl font-black text-white">Trading Hours Analysis</h3>
                <p className="text-neutral-400 text-sm">Performance by time of day</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {timeDistribution.map((slot, index) => {
                const maxTrades = Math.max(...timeDistribution.map(t => t.trades));
                const intensity = (slot.trades / maxTrades) * 100;

                return (
                  <div key={index} className="bg-white/5 rounded-xl p-4 border border-white/10 hover:border-primary-500/50 transition-all">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <FontAwesomeIcon icon={faClock} className="text-primary-400 text-sm" />
                        <span className="text-white font-bold">{slot.time}</span>
                      </div>
                      <span className={`text-xs font-bold px-2 py-1 rounded-lg ${
                        slot.winRate >= 70 ? 'bg-bull/20 text-bull' :
                        slot.winRate >= 65 ? 'bg-primary-500/20 text-primary-400' :
                        'bg-white/10 text-neutral-400'
                      }`}>
                        {slot.winRate.toFixed(1)}%
                      </span>
                    </div>
                    <div className="mb-2">
                      <div className="text-2xl font-black text-white">{slot.trades}</div>
                      <div className="text-neutral-500 text-xs">trades</div>
                    </div>
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden mb-2">
                      <div
                        className="h-full bg-gradient-to-r from-primary-500 to-purple-500 rounded-full"
                        style={{ width: `${intensity}%` }}
                      />
                    </div>
                    <div className="text-bull font-bold text-sm">
                      {formatCurrency(slot.profit)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
