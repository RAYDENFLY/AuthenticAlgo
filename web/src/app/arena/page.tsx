'use client';

import { useEffect, useRef, useState } from 'react';
import anime from 'animejs';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faTrophy,
  faRobot,
  faBrain,
  faChartLine,
  faPlay,
  faStop,
  faCrown,
  faFire,
  faDollarSign,
  faPercentage,
  faArrowTrendUp,
  faArrowTrendDown,
  faRocket,
  faShield,
  faBolt,
  faUsers,
  faMedal,
  faFlagCheckered,
  faChartBar,
  faRankingStar,
} from '@fortawesome/free-solid-svg-icons';
import { useCompetitionStatus } from '@/lib/hooks';
import { arenaAPI } from '@/lib/api';

export default function ArenaPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [initialCapital, setInitialCapital] = useState(10);
  const [maxTrades, setMaxTrades] = useState(10);
  const [isStarting, setIsStarting] = useState(false);
  
  const { data: competition, isLoading, mutate } = useCompetitionStatus();

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
    if (value > 0) return 'text-bull font-black';
    if (value < 0) return 'text-bear font-black';
    return 'text-neutral-300 font-black';
  };

  const handleStartCompetition = async () => {
    setIsStarting(true);
    try {
      await arenaAPI.startCompetition({
        initial_capital: initialCapital,
        max_trades: maxTrades,
        symbol: 'BTCUSDT',
        leverage_range: [1, 3],
      });
      mutate();
    } catch (error) {
      console.error('Failed to start competition:', error);
    } finally {
      setIsStarting(false);
    }
  };

  const getStrategyIcon = (strategy: string) => {
    switch (strategy) {
      case 'technical':
        return faChartLine;
      case 'ml':
        return faBrain;
      case 'hybrid':
        return faRobot;
      default:
        return faChartLine;
    }
  };

  const getStrategyColor = (strategy: string) => {
    switch (strategy) {
      case 'technical':
        return 'from-blue-500 to-cyan-500';
      case 'ml':
        return 'from-purple-500 to-pink-500';
      case 'hybrid':
        return 'from-orange-500 to-red-500';
      default:
        return 'from-gray-500 to-gray-600';
    }
  };

  const getStrategyGradient = (strategy: string) => {
    switch (strategy) {
      case 'technical':
        return 'bg-gradient-to-br from-blue-500/20 to-cyan-500/20 border-blue-500/30';
      case 'ml':
        return 'bg-gradient-to-br from-purple-500/20 to-pink-500/20 border-purple-500/30';
      case 'hybrid':
        return 'bg-gradient-to-br from-orange-500/20 to-red-500/20 border-orange-500/30';
      default:
        return 'bg-gradient-to-br from-gray-500/20 to-gray-600/20 border-gray-500/30';
    }
  };

  const strategies = competition?.strategies || [];
  const leader = strategies.length > 0 
    ? [...strategies].sort((a: any, b: any) => b.capital - a.capital)[0]
    : null;

  return (
    <div className="min-h-screen bg-dark-bg py-24 px-4">
      <div className="max-w-[1600px] mx-auto" ref={containerRef}>
        {/* Header */}
        <div className="mb-12 fade-in-card">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 bg-gradient-to-br from-yellow-500 to-orange-500 rounded-2xl flex items-center justify-center shadow-xl shadow-yellow-500/25">
              <FontAwesomeIcon icon={faTrophy} className="text-white text-2xl" />
            </div>
            <div>
              <h1 className="text-5xl font-black text-white mb-2">
                Trading Arena
              </h1>
              <p className="text-lg text-neutral-400">
                3-way strategy competition with real-time performance tracking
              </p>
            </div>
          </div>

          {/* Competition Status Badge */}
          {competition?.is_active && (
            <div className="inline-flex items-center gap-2 bg-bull/20 border border-bull/30 rounded-xl px-4 py-2">
              <div className="w-2 h-2 bg-bull rounded-full animate-pulse"></div>
              <span className="text-bull font-bold text-sm">LIVE BATTLE</span>
            </div>
          )}
        </div>

        {/* Competition Status */}
        {isLoading ? (
          <div className="text-center py-20">
            <div className="inline-block animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-primary-500 mb-6"></div>
            <p className="text-neutral-400 text-lg">Loading arena data...</p>
          </div>
        ) : competition ? (
          <>
            {/* Leader Banner */}
            {leader && (
              <div className="fade-in-card mb-12">
                <div className="bg-gradient-to-br from-yellow-500/20 to-orange-500/5 border border-yellow-500/30 rounded-3xl p-8">
                  <div className="flex flex-col md:flex-row items-center justify-between gap-6">
                    <div className="flex items-center gap-6">
                      <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-yellow-400 to-orange-500 flex items-center justify-center shadow-xl">
                        <FontAwesomeIcon icon={faCrown} className="text-white text-3xl" />
                      </div>
                      <div>
                        <div className="text-xs text-yellow-400 uppercase tracking-wider mb-2 font-bold flex items-center gap-2">
                          <FontAwesomeIcon icon={faTrophy} />
                          Current Leader
                        </div>
                        <h2 className="text-3xl font-black text-white capitalize mb-1">
                          {leader.name} Strategy
                        </h2>
                        <p className="text-neutral-400">
                          {leader.trades_count} trades • {leader.win_rate.toFixed(1)}% win rate
                        </p>
                      </div>
                    </div>
                    <div className="text-center md:text-right">
                      <div className="text-4xl font-black font-mono text-white mb-2">
                        {formatCurrency(leader.capital)}
                      </div>
                      <div className={`text-2xl font-black ${getPriceClass(leader.roi)}`}>
                        {formatPercentage(leader.roi)}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Strategy Cards */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-12">
              {strategies.map((strategy: any, index: number) => (
                <div 
                  key={strategy.name}
                  className="fade-in-card group"
                >
                  <div className={`bg-white/5 backdrop-blur-sm border rounded-3xl p-6 hover:border-primary-500/50 transition-all duration-300 h-full ${
                    index === 0 ? 'border-yellow-500/50 bg-yellow-500/5' : 'border-white/10'
                  }`}>
                    
                    {/* Position Badge */}
                    <div className="flex items-center justify-between mb-6">
                      <div className={`w-10 h-10 rounded-xl flex items-center justify-center font-black text-white text-sm ${
                        index === 0 ? 'bg-gradient-to-br from-yellow-500 to-orange-500' :
                        index === 1 ? 'bg-gradient-to-br from-gray-400 to-gray-500' :
                        'bg-gradient-to-br from-orange-600 to-red-600'
                      }`}>
                        #{index + 1}
                      </div>
                      {index === 0 && (
                        <FontAwesomeIcon icon={faCrown} className="text-yellow-400 text-xl" />
                      )}
                    </div>

                    {/* Header */}
                    <div className="flex items-center gap-3 mb-6">
                      <div className={`w-12 h-12 rounded-2xl bg-gradient-to-br ${getStrategyColor(strategy.name)} flex items-center justify-center`}>
                        <FontAwesomeIcon icon={getStrategyIcon(strategy.name)} className="text-white text-xl" />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-xl font-black text-white capitalize">{strategy.name}</h3>
                        <p className="text-neutral-500 text-sm">{strategy.description}</p>
                      </div>
                    </div>

                    {/* Capital */}
                    <div className="mb-6">
                      <div className="flex items-baseline justify-between mb-2">
                        <span className="text-neutral-400 text-sm font-semibold uppercase tracking-wider">
                          Capital
                        </span>
                        <span className="font-mono font-black text-2xl text-white">
                          {formatCurrency(strategy.capital)}
                        </span>
                      </div>
                      <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                        <div 
                          className={`h-full bg-gradient-to-r ${getStrategyColor(strategy.name)} rounded-full transition-all duration-1000`}
                          style={{ width: `${Math.min((strategy.capital / initialCapital) * 100, 200)}%` }}
                        />
                      </div>
                    </div>

                    {/* Stats Grid */}
                    <div className="grid grid-cols-2 gap-3 mb-6">
                      {[
                        { 
                          label: 'ROI', 
                          value: formatPercentage(strategy.roi), 
                          color: getPriceClass(strategy.roi)
                        },
                        { 
                          label: 'Trades', 
                          value: strategy.trades_count, 
                          color: 'text-white'
                        },
                        { 
                          label: 'Win Rate', 
                          value: `${strategy.win_rate.toFixed(1)}%`, 
                          color: strategy.win_rate >= 60 ? 'text-bull' : 'text-white'
                        },
                        { 
                          label: 'Leverage', 
                          value: `${strategy.avg_leverage.toFixed(1)}x`, 
                          color: 'text-primary-400'
                        }
                      ].map((stat, idx) => (
                        <div key={idx} className="bg-white/5 rounded-xl p-3 border border-white/10">
                          <div className="text-neutral-500 text-xs font-bold mb-1 uppercase">
                            {stat.label}
                          </div>
                          <div className={`font-mono font-bold ${stat.color}`}>
                            {stat.value}
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Recent Performance */}
                    <div className="pt-4 border-t border-white/10">
                      <div className="text-neutral-400 text-xs font-semibold mb-2 uppercase">
                        Recent Trades
                      </div>
                      <div className="flex gap-1">
                        {strategy.recent_trades.map((result: boolean, idx: number) => (
                          <div 
                            key={idx}
                            className={`flex-1 h-2 rounded-lg ${
                              result ? 'bg-bull' : 'bg-bear'
                            }`}
                          />
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Competition Controls & Info */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              {/* Competition Details */}
              <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-3xl p-8">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-12 h-12 bg-primary-500/20 rounded-2xl flex items-center justify-center">
                    <FontAwesomeIcon icon={faFlagCheckered} className="text-primary-400 text-xl" />
                  </div>
                  <h3 className="text-2xl font-black text-white">Battle Details</h3>
                </div>
                <div className="space-y-3">
                  {[
                    { label: 'Status', value: competition.is_active ? 'Active' : 'Completed', color: competition.is_active ? 'text-bull' : 'text-neutral-400' },
                    { label: 'Initial Capital', value: formatCurrency(competition.initial_capital), color: 'text-white' },
                    { label: 'Max Trades', value: competition.max_trades, color: 'text-white' },
                    { label: 'Symbol', value: competition.symbol, color: 'text-primary-400' },
                    { label: 'Duration', value: competition.elapsed_time ? `${competition.elapsed_time.toFixed(0)}s` : '0s', color: 'text-white' },
                    { label: 'Strategies', value: '3 AI Systems', color: 'text-purple-400' }
                  ].map((item, index) => (
                    <div key={index} className="flex items-center justify-between p-4 bg-white/5 rounded-xl border border-white/10">
                      <span className="text-neutral-400 font-semibold text-sm">{item.label}</span>
                      <span className={`font-mono font-bold ${item.color}`}>
                        {item.value}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Start Competition */}
              <div className="bg-gradient-to-br from-primary-500/20 to-purple-500/5 border border-primary-500/30 rounded-3xl p-8">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-12 h-12 bg-primary-500/20 rounded-2xl flex items-center justify-center">
                    <FontAwesomeIcon icon={faRocket} className="text-primary-400 text-xl" />
                  </div>
                  <h3 className="text-2xl font-black text-white">Launch Battle</h3>
                </div>
                <div className="space-y-4">
                  <div>
                    <label className="block text-neutral-400 text-xs font-bold mb-2 uppercase tracking-wider">
                      Initial Capital
                    </label>
                    <input
                      type="number"
                      value={initialCapital}
                      onChange={(e) => setInitialCapital(Number(e.target.value))}
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white font-mono focus:outline-none focus:border-primary-500 transition-all"
                      min="5"
                      max="1000"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-neutral-400 text-xs font-bold mb-2 uppercase tracking-wider">
                      Max Trades
                    </label>
                    <input
                      type="number"
                      value={maxTrades}
                      onChange={(e) => setMaxTrades(Number(e.target.value))}
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white font-mono focus:outline-none focus:border-primary-500 transition-all"
                      min="5"
                      max="100"
                    />
                  </div>

                  <button
                    onClick={handleStartCompetition}
                    disabled={isStarting || competition.is_active}
                    className="w-full bg-gradient-to-r from-primary-500 to-purple-600 hover:from-primary-600 hover:to-purple-700 text-white font-bold px-6 py-4 rounded-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <FontAwesomeIcon icon={isStarting ? faBolt : faRocket} className="mr-2" />
                    {isStarting ? 'Launching...' : competition.is_active ? 'Battle Active' : 'Start Battle'}
                  </button>

                  {competition.is_active && (
                    <div className="text-center p-3 bg-bull/20 border border-bull/30 rounded-xl">
                      <div className="flex items-center justify-center gap-2 text-bull font-bold text-sm">
                        <FontAwesomeIcon icon={faFire} />
                        Battle in progress!
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </>
        ) : (
          /* No Competition State */
          <div className="text-center py-24">
            <div className="w-24 h-24 bg-white/5 rounded-3xl flex items-center justify-center mx-auto mb-6">
              <FontAwesomeIcon icon={faTrophy} className="text-neutral-600 text-5xl" />
            </div>
            <h2 className="text-3xl font-black text-white mb-3">No Active Battle</h2>
            <p className="text-neutral-400 mb-12">Launch a competition to see strategies compete</p>
            
            {/* Start Competition Card */}
            <div className="max-w-2xl mx-auto">
              <div className="bg-gradient-to-br from-primary-500/20 to-purple-500/5 border border-primary-500/30 rounded-3xl p-8">
                <h3 className="text-2xl font-black text-white mb-6">Configure Battle</h3>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-neutral-400 text-xs font-bold mb-2 uppercase tracking-wider">
                        Initial Capital
                      </label>
                      <input
                        type="number"
                        value={initialCapital}
                        onChange={(e) => setInitialCapital(Number(e.target.value))}
                        className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white font-mono focus:outline-none focus:border-primary-500 transition-all"
                        min="5"
                        max="1000"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-neutral-400 text-xs font-bold mb-2 uppercase tracking-wider">
                        Max Trades
                      </label>
                      <input
                        type="number"
                        value={maxTrades}
                        onChange={(e) => setMaxTrades(Number(e.target.value))}
                        className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white font-mono focus:outline-none focus:border-primary-500 transition-all"
                        min="5"
                        max="100"
                      />
                    </div>
                  </div>

                  <button
                    onClick={handleStartCompetition}
                    disabled={isStarting}
                    className="w-full bg-gradient-to-r from-primary-500 to-purple-600 hover:from-primary-600 hover:to-purple-700 text-white font-bold text-lg px-8 py-4 rounded-xl transition-all duration-300"
                  >
                    <FontAwesomeIcon icon={faRocket} className="mr-2" />
                    {isStarting ? 'Launching...' : 'Start Battle'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}