'use client';

import { useEffect, useRef, useState } from 'react';
import anime from 'animejs';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faChartLine,
  faWallet,
  faTrophy,
  faArrowTrendUp,
  faArrowTrendDown,
  faCircle,
  faClock,
  faPercentage,
  faDollarSign,
  faChartPie,
  faBolt,
  faRocket,
  faShield,
  faFire,
  faCrown,
  faEye,
  faExchangeAlt,
  faHistory,
  faRobot,
} from '@fortawesome/free-solid-svg-icons';
import {
  usePublicStats,
  useLivePerformance,
  useCurrentPositions,
  useTradeHistory,
  usePortfolioBreakdown,
} from '@/lib/hooks';

export default function DashboardPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [activeTab, setActiveTab] = useState('overview');
  
  const { data: stats, isLoading: statsLoading } = usePublicStats();
  const { data: performance, isLoading: perfLoading } = useLivePerformance();
  const { data: positions, isLoading: positionsLoading } = useCurrentPositions();
  const { data: trades, isLoading: tradesLoading } = useTradeHistory(10);
  const { data: portfolio, isLoading: portfolioLoading } = usePortfolioBreakdown();

  useEffect(() => {
    if (containerRef.current) {
      anime.timeline({
        targets: containerRef.current.children,
      })
      .add({
        opacity: [0, 1],
        translateY: [50, 0],
        delay: anime.stagger(100),
        duration: 1000,
        easing: 'easeOutCubic',
      })
      .add({
        targets: '.stat-card',
        scale: [0.9, 1],
        opacity: [0, 1],
        delay: anime.stagger(80),
        duration: 800,
        easing: 'easeOutBack',
      }, '-=800');
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

  const getTrendIcon = (value: number) => {
    if (value > 0) return faArrowTrendUp;
    if (value < 0) return faArrowTrendDown;
    return faCircle;
  };

  return (
    <div className="min-h-screen bg-dark-bg py-24 px-4">
      <div className="mx-auto max-w-[1600px]" ref={containerRef}>
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 bg-gradient-to-br from-primary-500 to-purple-500 rounded-2xl flex items-center justify-center shadow-xl shadow-primary-500/25">
              <FontAwesomeIcon icon={faChartLine} className="text-white text-2xl" />
            </div>
            <div>
              <h1 className="text-5xl font-black text-white mb-2">
                Trading Dashboard
              </h1>
              <p className="text-lg text-neutral-400">
                Real-time performance & active positions
              </p>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex gap-2 bg-white/5 backdrop-blur-sm rounded-2xl p-2 border border-white/10 inline-flex">
            {['overview', 'positions', 'history', 'analytics'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-6 py-3 rounded-xl font-bold transition-all duration-300 ${
                  activeTab === tab
                    ? 'bg-primary-500 text-white shadow-lg shadow-primary-500/25'
                    : 'text-neutral-400 hover:text-white hover:bg-white/5'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {[
            {
              icon: faWallet,
              value: stats?.total_balance || 0,
              label: 'Total Balance',
              change: stats?.total_pnl_percentage || 0,
              color: 'from-primary-500 to-purple-500',
              iconBg: 'bg-primary-500/20',
              iconColor: 'text-primary-400',
              loading: statsLoading,
              trend: stats?.total_pnl_percentage || 0
            },
            {
              icon: faBolt,
              value: performance?.today_pnl || 0,
              label: "Today's P&L",
              change: performance?.today_pnl_pct || 0,
              color: 'from-bull to-bull-light',
              iconBg: 'bg-bull/20',
              iconColor: 'text-bull',
              loading: perfLoading,
              trend: performance?.today_pnl || 0
            },
            {
              icon: faTrophy,
              value: stats?.win_rate || 0,
              label: 'Win Rate',
              change: 0,
              color: 'from-purple-500 to-pink-500',
              iconBg: 'bg-purple-500/20',
              iconColor: 'text-purple-400',
              loading: statsLoading,
              suffix: '%',
              description: `${stats?.winning_trades || 0}W / ${stats?.losing_trades || 0}L`
            },
            {
              icon: faChartPie,
              value: positions?.length || 0,
              label: 'Active Positions',
              change: 0,
              color: 'from-orange-500 to-yellow-500',
              iconBg: 'bg-orange-500/20',
              iconColor: 'text-orange-400',
              loading: positionsLoading,
              description: `${formatCurrency(positions?.reduce((sum: number, p: any) => sum + p.size * p.entry_price, 0) || 0)}`
            }
          ].map((stat, index) => (
            <div key={index} className="stat-card">
              <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-3xl p-6 hover:border-primary-500/50 transition-all duration-300 group h-full">
                <div className="flex items-center justify-between mb-6">
                  <div className={`w-14 h-14 ${stat.iconBg} rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300`}>
                    <FontAwesomeIcon 
                      icon={stat.icon} 
                      className={`${stat.iconColor} text-2xl`} 
                    />
                  </div>
                  {stat.trend !== 0 && (
                    <div className={`px-3 py-1 rounded-xl ${stat.trend > 0 ? 'bg-bull/20 text-bull' : 'bg-bear/20 text-bear'} font-bold text-sm`}>
                      <FontAwesomeIcon icon={getTrendIcon(stat.trend)} className="mr-1" />
                      {formatPercentage(Math.abs(stat.trend))}
                    </div>
                  )}
                </div>
                
                <div className="mb-2">
                  <div className="text-neutral-400 text-sm font-semibold mb-2 uppercase tracking-wider">
                    {stat.label}
                  </div>
                  <div className="text-3xl font-black text-white">
                    {stat.loading ? (
                      <div className="h-8 w-24 bg-white/10 rounded-lg animate-pulse"></div>
                    ) : (
                      <>
                        {typeof stat.value === 'number' && stat.value >= 1000 ? formatCurrency(stat.value) : stat.value}
                        {stat.suffix}
                      </>
                    )}
                  </div>
                </div>
                
                {stat.description && (
                  <div className="text-sm text-neutral-500 mt-3 pt-3 border-t border-white/10">
                    {stat.loading ? '...' : stat.description}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Current Positions */}
        <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-3xl p-8 mb-8">
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-primary-500/20 rounded-2xl flex items-center justify-center">
                <FontAwesomeIcon icon={faExchangeAlt} className="text-primary-400 text-xl" />
              </div>
              <h2 className="text-3xl font-black text-white">Active Positions</h2>
            </div>
            <div className="flex items-center gap-3">
              <span className="px-4 py-2 bg-white/10 border border-white/20 rounded-xl text-neutral-300 font-bold">
                {positionsLoading ? '...' : `${positions?.length || 0} Active`}
              </span>
              {positions && positions.length > 0 && (
                <div className="flex items-center gap-2 px-4 py-2 bg-bull/20 border border-bull/30 rounded-xl">
                  <div className="w-2 h-2 bg-bull rounded-full animate-pulse"></div>
                  <span className="text-bull font-bold text-sm">LIVE</span>
                </div>
              )}
            </div>
          </div>

          {positionsLoading ? (
            <div className="text-center py-20">
              <div className="inline-block animate-spin rounded-full h-16 w-16 border-4 border-primary-500/20 border-t-primary-500 mb-4"></div>
              <p className="text-neutral-400 text-lg">Loading positions...</p>
            </div>
          ) : positions && positions.length > 0 ? (
            <div className="space-y-6">
              {positions.map((position: any) => {
                const unrealizedPnL = (position.current_price - position.entry_price) * position.size * (position.side === 'LONG' ? 1 : -1);
                const unrealizedPnLPct = ((position.current_price - position.entry_price) / position.entry_price) * 100 * (position.side === 'LONG' ? 1 : -1);
                
                return (
                  <div key={position.id} className="bg-white/5 border border-white/10 rounded-3xl p-6 hover:border-primary-500/50 transition-all duration-300">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center gap-4">
                        <div className={`w-3 h-3 rounded-full ${position.side === 'LONG' ? 'bg-bull' : 'bg-bear'} shadow-lg animate-pulse`} />
                        <div>
                          <h3 className="text-2xl font-black text-white mb-2">{position.symbol}</h3>
                          <span className={`inline-block px-4 py-2 rounded-xl font-bold text-sm border-2 ${
                            position.side === 'LONG' 
                              ? 'bg-bull/20 text-bull border-bull/30' 
                              : 'bg-bear/20 text-bear border-bear/30'
                          }`}>
                            {position.side} • {position.leverage}x
                          </span>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`text-3xl font-black font-mono mb-1 ${getPriceClass(unrealizedPnL)}`}>
                          {formatCurrency(unrealizedPnL)}
                        </div>
                        <div className={`text-lg font-bold ${getPriceClass(unrealizedPnLPct)}`}>
                          {formatPercentage(unrealizedPnLPct)}
                        </div>
                      </div>
                    </div>

                    {/* Price Grid */}
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                      {[
                        { label: 'Entry Price', value: position.entry_price, icon: faCircle },
                        { label: 'Current Price', value: position.current_price, icon: faChartLine },
                        { label: 'Size', value: position.size.toFixed(4), icon: faChartPie, suffix: '' },
                        { label: 'Value', value: position.size * position.entry_price, icon: faDollarSign }
                      ].map((item, idx) => (
                        <div key={idx} className="bg-white/5 border border-white/10 rounded-2xl p-4 hover:bg-white/10 transition-colors duration-300">
                          <div className="flex items-center gap-2 mb-2">
                            <FontAwesomeIcon icon={item.icon} className="text-primary-400 text-xs" />
                            <span className="text-neutral-400 text-xs font-bold uppercase tracking-wider">
                              {item.label}
                            </span>
                          </div>
                          <div className="font-mono font-black text-xl text-white">
                            {typeof item.value === 'number' && item.value >= 100 ? formatCurrency(item.value) : item.value}
                            {item.suffix}
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* TP/SL Levels */}
                    <div className="pt-6 border-t border-white/10">
                      <div className="flex items-center gap-2 mb-4">
                        <FontAwesomeIcon icon={faShield} className="text-primary-400" />
                        <span className="text-neutral-300 font-bold">Risk Management</span>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {[
                          { label: 'TP1', value: position.tp_levels[0], type: 'tp' },
                          { label: 'TP2', value: position.tp_levels[1], type: 'tp' },
                          { label: 'TP3', value: position.tp_levels[2], type: 'tp' },
                          { label: 'Stop Loss', value: position.stop_loss, type: 'sl' }
                        ].map((level, idx) => (
                          <div key={idx} className={`bg-white/5 border-2 rounded-xl p-3 text-center hover:scale-105 transition-transform ${
                            level.type === 'tp' ? 'border-bull/30 hover:border-bull/50' : 'border-bear/30 hover:border-bear/50'
                          }`}>
                            <div className="text-neutral-400 text-xs font-bold mb-1 uppercase tracking-wider">
                              {level.label}
                            </div>
                            <div className={`font-mono font-black text-sm ${
                              level.type === 'tp' ? 'text-bull' : 'text-bear'
                            }`}>
                              {formatCurrency(level.value)}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-24">
              <div className="w-24 h-24 bg-white/5 rounded-3xl flex items-center justify-center mx-auto mb-6">
                <FontAwesomeIcon icon={faChartLine} className="text-neutral-600 text-5xl" />
              </div>
              <h3 className="text-2xl font-bold text-white mb-3">No Active Positions</h3>
              <p className="text-neutral-400 mb-6">Waiting for AI trading signals...</p>
              <div className="flex items-center justify-center gap-3">
                <FontAwesomeIcon icon={faRobot} className="text-primary-400 animate-pulse" />
                <span className="text-neutral-500 font-semibold">AI System Monitoring Markets</span>
                <FontAwesomeIcon icon={faRobot} className="text-primary-400 animate-pulse" />
              </div>
            </div>
          )}
        </div>

        {/* Recent Trades */}
        <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-3xl p-8">
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-primary-500/20 rounded-2xl flex items-center justify-center">
                <FontAwesomeIcon icon={faHistory} className="text-primary-400 text-xl" />
              </div>
              <h2 className="text-3xl font-black text-white">Recent Trades</h2>
            </div>
            <span className="px-4 py-2 bg-white/10 border border-white/20 rounded-xl text-neutral-300 font-bold">
              {tradesLoading ? '...' : `Last ${trades?.trades.length || 0} Trades`}
            </span>
          </div>

          {tradesLoading ? (
            <div className="text-center py-20">
              <div className="inline-block animate-spin rounded-full h-16 w-16 border-4 border-primary-500/20 border-t-primary-500 mb-4"></div>
              <p className="text-neutral-400 text-lg">Loading trade history...</p>
            </div>
          ) : trades && trades.trades.length > 0 ? (
            <div className="overflow-hidden">
              <div className="space-y-4">
                {trades.trades.map((trade: any) => (
                  <div 
                    key={trade.id} 
                    className="bg-white/5 border border-white/10 rounded-2xl p-6 hover:border-primary-500/50 hover:bg-white/10 transition-all duration-300 group"
                  >
                    <div className="flex items-center justify-between">
                      {/* Left: Symbol & Side */}
                      <div className="flex items-center gap-4">
                        <div className={`w-3 h-3 rounded-full ${trade.side === 'LONG' ? 'bg-bull' : 'bg-bear'}`} />
                        <div>
                          <div className="font-black text-xl text-white mb-1">{trade.symbol}</div>
                          <span className={`inline-block px-3 py-1 rounded-lg font-bold text-xs border ${
                            trade.side === 'LONG' 
                              ? 'bg-bull/20 text-bull border-bull/30' 
                              : 'bg-bear/20 text-bear border-bear/30'
                          }`}>
                            {trade.side}
                          </span>
                        </div>
                      </div>

                      {/* Center: Prices */}
                      <div className="hidden md:flex items-center gap-8">
                        <div className="text-center">
                          <div className="text-neutral-500 text-xs font-bold mb-1 uppercase">Entry</div>
                          <div className="font-mono font-bold text-white">{formatCurrency(trade.entry_price)}</div>
                        </div>
                        <FontAwesomeIcon icon={faArrowTrendUp} className="text-neutral-600" />
                        <div className="text-center">
                          <div className="text-neutral-500 text-xs font-bold mb-1 uppercase">Exit</div>
                          <div className="font-mono font-bold text-white">
                            {trade.exit_price ? formatCurrency(trade.exit_price) : '-'}
                          </div>
                        </div>
                      </div>

                      {/* Right: P&L & Result */}
                      <div className="text-right">
                        <div className={`font-mono font-black text-2xl mb-1 ${getPriceClass(trade.pnl || 0)}`}>
                          {trade.pnl ? formatCurrency(trade.pnl) : '-'}
                        </div>
                        <div className="flex items-center justify-end gap-2">
                          <div className={`text-sm font-bold ${getPriceClass(trade.pnl_percentage || 0)}`}>
                            {trade.pnl_percentage ? formatPercentage(trade.pnl_percentage) : '-'}
                          </div>
                          {trade.exit_reason && (
                            <span className={`px-3 py-1 rounded-lg font-bold text-xs border ${
                              trade.exit_reason.includes('TP') 
                                ? 'bg-bull/20 text-bull border-bull/30' 
                                : trade.exit_reason === 'SL'
                                ? 'bg-bear/20 text-bear border-bear/30'
                                : 'bg-white/10 text-neutral-300 border-white/20'
                            }`}>
                              {trade.exit_reason.includes('TP') && <FontAwesomeIcon icon={faTrophy} className="mr-1" />}
                              {trade.exit_reason}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Mobile Price Info */}
                    <div className="md:hidden mt-4 pt-4 border-t border-white/10 grid grid-cols-2 gap-4">
                      <div>
                        <div className="text-neutral-500 text-xs font-bold mb-1 uppercase">Entry</div>
                        <div className="font-mono font-bold text-white">{formatCurrency(trade.entry_price)}</div>
                      </div>
                      <div>
                        <div className="text-neutral-500 text-xs font-bold mb-1 uppercase">Exit</div>
                        <div className="font-mono font-bold text-white">
                          {trade.exit_price ? formatCurrency(trade.exit_price) : '-'}
                        </div>
                      </div>
                    </div>

                    {/* Time */}
                    <div className="mt-4 flex items-center gap-2 text-neutral-500 text-sm">
                      <FontAwesomeIcon icon={faClock} className="text-xs" />
                      {new Date(trade.entry_time).toLocaleString()}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-center py-24">
              <div className="w-24 h-24 bg-white/5 rounded-3xl flex items-center justify-center mx-auto mb-6">
                <FontAwesomeIcon icon={faHistory} className="text-neutral-600 text-5xl" />
              </div>
              <h3 className="text-2xl font-bold text-white mb-3">No Trade History</h3>
              <p className="text-neutral-400">Your trading history will appear here</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}