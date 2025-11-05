'use client';

import { useEffect, useRef, useState } from 'react';
import anime from 'animejs';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faChartLine, 
  faRobot, 
  faTrophy, 
  faBolt,
  faShield,
  faChartBar,
  faRocket,
  faUsers,
  faGem,
  faPlay,
  faPause,
  faArrowUp,
  faArrowDown,
  faClock,
  faCode,
  faHeart,
  faGift
} from '@fortawesome/free-solid-svg-icons';
import { faDiscord, faGithub } from '@fortawesome/free-brands-svg-icons';
import Link from 'next/link';

export default function HomePage() {
  const heroRef = useRef(null);
  const statsRef = useRef(null);
  const [liveData, setLiveData] = useState({
    pnl: '+$45.23',
    winRate: '85.5%',
    totalTrades: '156',
    roi: '+452%'
  });

  // Live trading data
  const [liveTrades, setLiveTrades] = useState([
    { id: 1, strategy: 'Pure ML', side: 'LONG', entry: 43250, current: 43580, pnl: 7.63, roi: 2.8, time: '2m ago', status: 'active' },
    { id: 2, strategy: 'Hybrid', side: 'SHORT', entry: 43600, current: 43520, pnl: 1.84, roi: 0.7, time: '5m ago', status: 'active' },
    { id: 3, strategy: 'Technical', side: 'LONG', entry: 43100, current: 43200, pnl: 2.32, roi: 0.9, time: '8m ago', status: 'closed' },
  ]);

  const [botCapital, setBotCapital] = useState({
    initial: 5.00,
    current: 16.79,
    profit: 11.79,
    profitPercent: 235.8,
    trades: 34,
    winRate: 88.2
  });

  // Simulate live data updates
  useEffect(() => {
    const interval = setInterval(() => {
      setLiveData(prev => ({
        ...prev,
        pnl: `+$${(45.23 + Math.random() * 5).toFixed(2)}`,
        totalTrades: (156 + Math.floor(Math.random() * 3)).toString()
      }));

      // Update bot capital
      setBotCapital(prev => {
        const newCurrent = prev.current + (Math.random() - 0.3) * 0.5;
        const newProfit = newCurrent - prev.initial;
        return {
          ...prev,
          current: Math.max(prev.initial, newCurrent),
          profit: newProfit,
          profitPercent: (newProfit / prev.initial) * 100
        };
      });

      // Update live trades
      setLiveTrades(prev => prev.map(trade => ({
        ...trade,
        current: trade.current + (Math.random() - 0.5) * 50,
        pnl: trade.pnl + (Math.random() - 0.3) * 0.5,
        roi: trade.roi + (Math.random() - 0.3) * 0.2
      })));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Hero animation with more sophisticated effects
    if (heroRef.current) {
      anime.timeline({
        targets: heroRef.current,
      })
      .add({
        opacity: [0, 1],
        translateY: [50, 0],
        duration: 1200,
        easing: 'easeOutExpo',
      })
      .add({
        targets: '.hero-stats',
        opacity: [0, 1],
        translateY: [30, 0],
        delay: anime.stagger(100),
        duration: 800,
        easing: 'easeOutExpo',
      }, '-=600');
    }

    // Floating particles background
    anime({
      targets: '.floating-particle',
      translateX: () => anime.random(-100, 100),
      translateY: () => anime.random(-100, 100),
      duration: () => anime.random(2000, 5000),
      easing: 'easeInOutSine',
      loop: true,
      direction: 'alternate',
    });

    // Feature cards staggered entrance
    anime({
      targets: '.feature-card',
      opacity: [0, 1],
      translateY: [40, 0],
      delay: anime.stagger(150),
      duration: 800,
      easing: 'easeOutBack',
    });
  }, []);

  const features = [
    {
      icon: faRobot,
      title: "AI-Powered Intelligence",
      description: "Advanced machine learning models processing 1M+ data points in real-time",
      badges: ["96% Accuracy", "52 Features"],
      color: "primary"
    },
    {
      icon: faShield,
      title: "Smart Risk Management", 
      description: "Multi-layer protection with dynamic position sizing and real-time monitoring",
      badges: ["3 TP Levels", "ATR-Based SL"],
      color: "bull"
    },
    {
      icon: faBolt,
      title: "Lightning Execution",
      description: "Sub-100ms trade execution with institutional-grade infrastructure",
      badges: ["24/7 Active", "99.9% Uptime"],
      color: "purple"
    },
    {
      icon: faChartBar,
      title: "Live Performance Analytics",
      description: "Real-time dashboard with advanced metrics and strategy comparison",
      badges: ["Live P&L", "Risk Metrics"],
      color: "primary"
    },
    {
      icon: faUsers,
      title: "Multi-Strategy Portfolio",
      description: "Diversified approach combining ML, Technical Analysis, and Market Making",
      badges: ["3 Strategies", "Auto-Rebalancing"],
      color: "bull"
    },
    {
      icon: faGem,
      title: "Institutional Tools",
      description: "Professional-grade backtesting, paper trading, and risk analysis",
      badges: ["Backtesting", "Monte Carlo"],
      color: "purple"
    }
  ];

  return (
    <div className="min-h-screen bg-dark-bg overflow-hidden">
      {/* Animated Background Particles */}
      <div className="fixed inset-0 pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="floating-particle absolute w-1 h-1 bg-primary-400/20 rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
          />
        ))}
      </div>

      {/* Hero Section */}
      <section ref={heroRef} className="relative py-32 px-4 overflow-hidden bg-dark-bg">
        <div className="absolute inset-0 bg-gradient-to-br from-primary-900/10 via-transparent to-purple-900/10" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-primary-500/5 via-transparent to-transparent" />
        
        <div className="container mx-auto relative z-10 max-w-[1400px]">
          <div className="text-center">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 bg-primary-500/10 border border-primary-500/30 rounded-full px-4 py-2 mb-8">
              <div className="w-2 h-2 bg-bull rounded-full animate-pulse"></div>
              <span className="text-sm font-semibold text-primary-400">
                🚀 LIVE TRADING ACTIVE • +$45.23 TODAY
              </span>
            </div>

            <h1 className="text-6xl md:text-8xl font-black mb-8 leading-tight bg-gradient-to-r from-primary-400 via-purple-400 to-bull bg-clip-text text-transparent">
              Trade Smarter
              <br />
              <span className="text-white">With AI Power</span>
            </h1>
            
            <p className="text-2xl md:text-3xl text-neutral-300 mb-12 mx-auto leading-relaxed max-w-5xl">
              Transform <span className="text-bull font-semibold">$5 into $100+</span> using our 
              advanced machine learning algorithms with{' '}
              <span className="text-primary-400 font-semibold">96% proven accuracy</span>
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-6 justify-center items-center mb-16">
              <Link 
                href="/reports" 
                className="group bg-gradient-to-r from-primary-500 to-purple-500 hover:from-primary-600 hover:to-purple-600 text-white font-bold text-lg px-10 py-4 rounded-2xl transition-all duration-300 transform hover:scale-105 shadow-xl shadow-primary-500/25 inline-flex items-center gap-3"
              >
                <FontAwesomeIcon icon={faChartBar} />
                View Performance Reports
              </Link>
              
              <Link 
                href="/documentation" 
                className="group border-2 border-white/20 hover:border-primary-400 text-white hover:text-primary-400 font-bold text-lg px-10 py-4 rounded-2xl transition-all duration-300 hover:bg-white/5 inline-flex items-center gap-3"
              >
                <FontAwesomeIcon icon={faRocket} />
                Read Documentation
              </Link>
            </div>

            {/* Live Stats Grid */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 mx-auto mb-16 max-w-[1200px]">
              {[
                { value: liveData.pnl, label: 'Live P&L', trend: 'up' },
                { value: liveData.winRate, label: 'Win Rate', trend: 'neutral' },
                { value: liveData.totalTrades, label: 'Total Trades', trend: 'neutral' },
                { value: liveData.roi, label: 'Total ROI', trend: 'up' }
              ].map((stat, index) => (
                <div 
                  key={index}
                  className="hero-stats group bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-6 hover:bg-white/10 transition-all duration-300 hover:scale-105"
                >
                  <div className={`text-2xl font-bold mb-2 ${
                    stat.trend === 'up' ? 'text-bull' : 'text-white'
                  }`}>
                    {stat.value}
                  </div>
                  <div className="text-sm text-neutral-400 font-medium">
                    {stat.label}
                  </div>
                </div>
              ))}
            </div>

            {/* Bot Capital Progress */}
            <div className="mx-auto mb-12 max-w-[1200px]">
              <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-3xl p-8">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h3 className="text-2xl font-bold text-white mb-1">Live Bot Performance</h3>
                    <p className="text-neutral-400">Real-time capital growth tracker</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-bull rounded-full animate-pulse"></div>
                    <span className="text-bull font-semibold">ACTIVE</span>
                  </div>
                </div>

                {/* Capital Stats */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
                  <div>
                    <div className="text-neutral-400 text-sm mb-1">Initial Capital</div>
                    <div className="text-2xl font-bold text-white">${botCapital.initial.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-neutral-400 text-sm mb-1">Current Capital</div>
                    <div className="text-2xl font-bold text-bull">${botCapital.current.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-neutral-400 text-sm mb-1">Total Profit</div>
                    <div className="text-2xl font-bold text-bull">+${botCapital.profit.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-neutral-400 text-sm mb-1">ROI</div>
                    <div className="text-2xl font-bold text-primary-400">+{botCapital.profitPercent.toFixed(1)}%</div>
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="mb-8">
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-neutral-400">Progress to $100</span>
                    <span className="text-bull font-semibold">{((botCapital.current / 100) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-4 bg-dark-border rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-bull to-primary-500 transition-all duration-500"
                      style={{ width: `${Math.min((botCapital.current / 100) * 100, 100)}%` }}
                    ></div>
                  </div>
                  <div className="flex justify-between text-xs mt-2 text-neutral-500">
                    <span>$5</span>
                    <span>$100 Target</span>
                  </div>
                </div>

                {/* Additional Stats */}
                <div className="grid grid-cols-2 gap-4 pt-6 border-t border-white/10">
                  <div className="flex items-center justify-between">
                    <span className="text-neutral-400">Total Trades</span>
                    <span className="text-white font-semibold">{botCapital.trades}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-neutral-400">Win Rate</span>
                    <span className="text-bull font-semibold">{botCapital.winRate.toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Live Trading Table */}
            <div className="mx-auto max-w-[1400px]">
              <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-3xl p-8">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-2xl font-bold text-white">Live Trades</h3>
                  <div className="flex items-center gap-2 text-bull text-sm font-semibold">
                    <div className="w-2 h-2 bg-bull rounded-full animate-pulse"></div>
                    STREAMING
                  </div>
                </div>

                {/* Table */}
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-white/10">
                        <th className="text-left py-3 px-4 text-neutral-400 text-sm font-semibold">Strategy</th>
                        <th className="text-left py-3 px-4 text-neutral-400 text-sm font-semibold">Side</th>
                        <th className="text-right py-3 px-4 text-neutral-400 text-sm font-semibold">Entry</th>
                        <th className="text-right py-3 px-4 text-neutral-400 text-sm font-semibold">Current</th>
                        <th className="text-right py-3 px-4 text-neutral-400 text-sm font-semibold">P&L</th>
                        <th className="text-right py-3 px-4 text-neutral-400 text-sm font-semibold">ROI</th>
                        <th className="text-center py-3 px-4 text-neutral-400 text-sm font-semibold">Time</th>
                        <th className="text-center py-3 px-4 text-neutral-400 text-sm font-semibold">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {liveTrades.map((trade) => (
                        <tr key={trade.id} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                          <td className="py-4 px-4">
                            <span className="font-semibold text-white">{trade.strategy}</span>
                          </td>
                          <td className="py-4 px-4">
                            <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                              trade.side === 'LONG' 
                                ? 'bg-bull/20 text-bull border border-bull/30' 
                                : 'bg-bear/20 text-bear border border-bear/30'
                            }`}>
                              {trade.side}
                            </span>
                          </td>
                          <td className="py-4 px-4 text-right font-mono text-white">
                            ${trade.entry.toLocaleString()}
                          </td>
                          <td className="py-4 px-4 text-right font-mono text-white">
                            ${trade.current.toLocaleString()}
                          </td>
                          <td className="py-4 px-4 text-right">
                            <span className={`font-mono font-bold ${trade.pnl > 0 ? 'text-bull' : 'text-bear'}`}>
                              {trade.pnl > 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                            </span>
                          </td>
                          <td className="py-4 px-4 text-right">
                            <span className={`font-mono font-bold ${trade.roi > 0 ? 'text-bull' : 'text-bear'}`}>
                              {trade.roi > 0 ? '+' : ''}{trade.roi.toFixed(1)}%
                            </span>
                          </td>
                          <td className="py-4 px-4 text-center">
                            <div className="flex items-center justify-center gap-2 text-neutral-400 text-sm">
                              <FontAwesomeIcon icon={faClock} className="text-xs" />
                              {trade.time}
                            </div>
                          </td>
                          <td className="py-4 px-4 text-center">
                            <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                              trade.status === 'active'
                                ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30'
                                : 'bg-neutral/20 text-neutral border border-neutral/30'
                            }`}>
                              {trade.status.toUpperCase()}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Supported Exchanges Section */}
      <section className="py-24 px-4 bg-dark-bg border-t border-white/10">
        <div className="container mx-auto max-w-[1400px]">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-black mb-6">
              <span className="bg-gradient-to-r from-primary-400 to-purple-400 bg-clip-text text-transparent">
                Supported Exchanges
              </span>
            </h2>
            <p className="text-xl text-neutral-400">
              Trade seamlessly across multiple exchanges with unified API integration
            </p>
          </div>

          {/* Active Exchanges */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mx-auto mb-12 max-w-[1200px]">
            {/* Binance Card */}
            <div className="group bg-gradient-to-br from-[#F3BA2F]/10 to-[#F3BA2F]/5 border border-[#F3BA2F]/30 rounded-3xl p-8 hover:border-[#F3BA2F]/50 hover:scale-105 transition-all duration-300">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-20 h-20 bg-[#F3BA2F]/20 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform">
                  <span className="text-4xl font-black text-[#F3BA2F]">B</span>
                </div>
                <div>
                  <h3 className="text-3xl font-black text-white">Binance</h3>
                  <p className="text-neutral-400">World's Leading Exchange</p>
                </div>
              </div>
              
              <ul className="space-y-3 mb-6">
                <li className="flex items-start gap-3">
                  <div className="w-1.5 h-1.5 bg-[#F3BA2F] rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-neutral-300">Spot & Futures Trading</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-1.5 h-1.5 bg-[#F3BA2F] rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-neutral-300">Real-time WebSocket Streams</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-1.5 h-1.5 bg-[#F3BA2F] rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-neutral-300">Advanced Order Types</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-1.5 h-1.5 bg-[#F3BA2F] rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-neutral-300">High Liquidity & Low Fees</span>
                </li>
              </ul>

              <div className="flex gap-2 flex-wrap">
                <span className="px-4 py-2 bg-[#F3BA2F]/20 border border-[#F3BA2F]/30 rounded-xl text-sm font-bold text-[#F3BA2F]">
                  ✓ Production Ready
                </span>
                <span className="px-4 py-2 bg-bull/20 border border-bull/30 rounded-xl text-sm font-bold text-bull">
                  ✓ Active
                </span>
              </div>
            </div>

            {/* AsterDEX Card */}
            <div className="group bg-gradient-to-br from-primary-500/10 to-purple-500/10 border border-primary-500/30 rounded-3xl p-8 hover:border-primary-500/50 hover:scale-105 transition-all duration-300">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-20 h-20 bg-gradient-to-br from-primary-500/20 to-purple-500/20 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform">
                  <span className="text-4xl font-black text-primary-400">A</span>
                </div>
                <div>
                  <h3 className="text-3xl font-black text-white">AsterDEX</h3>
                  <p className="text-neutral-400">Decentralized Exchange</p>
                </div>
              </div>
              
              <ul className="space-y-3 mb-6">
                <li className="flex items-start gap-3">
                  <div className="w-1.5 h-1.5 bg-primary-400 rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-neutral-300">DEX Trading Integration</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-1.5 h-1.5 bg-primary-400 rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-neutral-300">Real-time Market Data API</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-1.5 h-1.5 bg-primary-400 rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-neutral-300">Smart Contract Automation</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-1.5 h-1.5 bg-primary-400 rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-neutral-300">On-chain Trading Execution</span>
                </li>
              </ul>

              <div className="flex gap-2 flex-wrap">
                <span className="px-4 py-2 bg-primary-500/20 border border-primary-500/30 rounded-xl text-sm font-bold text-primary-400">
                  ✓ Integrated
                </span>
                <span className="px-4 py-2 bg-bull/20 border border-bull/30 rounded-xl text-sm font-bold text-bull">
                  ✓ Active
                </span>
              </div>
            </div>
          </div>

          {/* Coming Soon */}
          <div className="text-center mx-auto max-w-[1000px]">
            <h3 className="text-2xl font-bold text-white mb-6">Coming Soon</h3>
            <div className="flex flex-wrap gap-4 justify-center">
              {['Bybit', 'OKX', 'Gate.io', 'KuCoin', 'Kraken', 'Huobi'].map((exchange) => (
                <div 
                  key={exchange}
                  className="px-6 py-3 bg-white/5 border border-white/10 rounded-xl text-neutral-400 hover:border-primary-500/30 hover:text-white transition-all duration-300"
                >
                  {exchange}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Community Section */}
      <section className="py-32 px-4 relative bg-dark-bg">
        <div className="container mx-auto">
          <div className="text-center mb-20">
            <h2 className="text-5xl md:text-6xl font-black mb-6">
              <span className="bg-gradient-to-r from-primary-400 to-purple-400 bg-clip-text text-transparent">
                Join Our
              </span>
              <br />
              <span className="text-white">Trading Community</span>
            </h2>
            <p className="text-xl text-neutral-400 max-w-2xl mx-auto">
              Connect with thousands of traders, share strategies, and grow together
            </p>
          </div>

          <div className="max-w-4xl mx-auto">
            {/* Discord CTA Card */}
            <div className="bg-gradient-to-br from-primary-900/30 to-purple-900/30 border border-primary-500/30 rounded-3xl p-12 text-center backdrop-blur-sm">
              <div className="w-20 h-20 bg-[#5865F2]/10 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <FontAwesomeIcon icon={faDiscord} className="text-5xl text-[#5865F2]" />
              </div>
              
              <h3 className="text-4xl font-black text-white mb-4">
                Join Our Discord Server
              </h3>
              
              <p className="text-xl text-neutral-300 mb-8 max-w-2xl mx-auto">
                Get real-time trading signals, discuss strategies with experts, and access exclusive resources
              </p>

              {/* Benefits Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
                {[
                  { icon: faUsers, title: '5,000+ Members', desc: 'Active trading community' },
                  { icon: faChartLine, title: 'Live Signals', desc: '24/7 real-time alerts' },
                  { icon: faRobot, title: 'Bot Updates', desc: 'Latest features & tips' }
                ].map((benefit, index) => (
                  <div key={index} className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-6">
                    <FontAwesomeIcon icon={benefit.icon} className="text-3xl text-primary-400 mb-3" />
                    <h4 className="text-lg font-bold text-white mb-2">{benefit.title}</h4>
                    <p className="text-sm text-neutral-400">{benefit.desc}</p>
                  </div>
                ))}
              </div>

              {/* Discord Button */}
              <a 
                href="https://discord.gg/your-server-link" 
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-3 bg-[#5865F2] hover:bg-[#4752C4] text-white font-bold text-xl px-12 py-5 rounded-2xl transition-all duration-300 transform hover:scale-105 shadow-2xl shadow-[#5865F2]/25"
              >
                <FontAwesomeIcon icon={faDiscord} className="text-2xl" />
                Join Discord Community
                <FontAwesomeIcon icon={faArrowUp} className="rotate-45" />
              </a>

              <p className="text-sm text-neutral-500 mt-6">
                Free to join • No credit card required
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Performance Section */}
      <section className="py-32 px-4 bg-dark-bg relative">
        <div className="absolute inset-0 bg-gradient-to-br from-primary-900/10 to-purple-900/10"></div>
        <div className="container mx-auto relative z-10">
          <div className="text-center mb-16">
            <h2 className="text-5xl md:text-6xl font-black mb-6">
              <span className="text-white">Proven</span>
              <span className="bg-gradient-to-r from-bull to-primary-400 bg-clip-text text-transparent"> Performance</span>
            </h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            {/* Left - Performance Metrics */}
            <div className="space-y-8">
              {[
                { metric: '96.2%', label: 'Model Accuracy', description: 'Backtested on 2+ years of market data' },
                { metric: '87.5%', label: 'Live Accuracy', description: 'Real-time trading performance' },
                { metric: '2.15', label: 'Sharpe Ratio', description: 'Risk-adjusted returns' },
                { metric: '-8.5%', label: 'Max Drawdown', description: 'Worst-case scenario protection' }
              ].map((item, index) => (
                <div key={index} className="flex items-center gap-6 p-6 bg-white/5 rounded-2xl border border-white/10">
                  <div className="text-3xl font-black text-primary-400 min-w-20">
                    {item.metric}
                  </div>
                  <div>
                    <div className="text-xl font-bold text-white mb-1">
                      {item.label}
                    </div>
                    <div className="text-neutral-400 text-sm">
                      {item.description}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Right - Strategy Comparison */}
            <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-3xl p-8">
              <h3 className="text-2xl font-bold text-center mb-8">Live Strategy Battle</h3>
              <div className="space-y-6">
                {[
                  { name: 'Pure ML', roi: '+52%', winRate: '90%', trades: 10, leading: true },
                  { name: 'Hybrid AI', roi: '+38%', winRate: '89%', trades: 9, leading: false },
                  { name: 'Technical', roi: '+25%', winRate: '75%', trades: 8, leading: false }
                ].map((strategy, index) => (
                  <div
                    key={index}
                    className={`p-6 rounded-2xl border-2 transition-all duration-300 ${
                      strategy.leading 
                        ? 'border-primary-500 bg-primary-500/10' 
                        : 'border-white/10 bg-white/5'
                    }`}
                  >
                    <div className="flex justify-between items-center mb-3">
                      <span className="text-lg font-bold text-white">
                        {strategy.name}
                      </span>
                      {strategy.leading && (
                        <span className="bg-bull text-white px-3 py-1 rounded-full text-sm font-bold">
                          LEADING
                        </span>
                      )}
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-neutral-400">ROI: <span className="text-bull font-semibold">{strategy.roi}</span></span>
                      <span className="text-neutral-400">Win Rate: <span className="text-primary-400 font-semibold">{strategy.winRate}</span></span>
                      <span className="text-neutral-400">Trades: {strategy.trades}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="py-32 px-4 bg-dark-bg">
        <div className="container mx-auto text-center">
          <div className="max-w-4xl mx-auto relative">
            {/* Glowing background effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-primary-500/20 to-purple-500/20 blur-3xl rounded-3xl" />
            
            <div className="relative bg-gradient-to-br from-primary-900/30 to-purple-900/30 border border-primary-500/30 rounded-3xl p-16 backdrop-blur-sm">
              <h2 className="text-5xl md:text-6xl font-black mb-8">
                <span className="bg-gradient-to-r from-primary-400 to-bull bg-clip-text text-transparent">
                  Ready to Join?
                </span>
              </h2>
              
              <p className="text-2xl text-neutral-300 mb-12 max-w-2xl mx-auto">
                Be part of our growing community of successful algo traders
              </p>

              <div className="flex flex-col sm:flex-row gap-6 justify-center items-center">
                <a 
                  href="https://discord.gg/your-server-link"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group bg-[#5865F2] hover:bg-[#4752C4] text-white font-bold text-xl px-16 py-5 rounded-2xl transition-all duration-300 transform hover:scale-105 shadow-2xl shadow-[#5865F2]/25 inline-flex items-center gap-3"
                >
                  <FontAwesomeIcon icon={faDiscord} className="text-2xl" />
                  Join Discord Now
                </a>
                
                <Link 
                  href="/reports" 
                  className="border-2 border-white/20 hover:border-primary-400 text-white hover:text-primary-400 font-bold text-xl px-12 py-5 rounded-2xl transition-all duration-300 hover:bg-white/5 inline-flex items-center gap-3"
                >
                  <FontAwesomeIcon icon={faChartBar} />
                  View Reports
                </Link>
              </div>

              {/* Trust badges */}
              <div className="flex justify-center items-center gap-8 mt-12 pt-12 border-t border-white/10">
                {['Open Source', 'Free to Join', 'Active Community'].map((badge, index) => (
                  <div key={index} className="flex items-center gap-2 text-neutral-400">
                    <div className="w-2 h-2 bg-bull rounded-full"></div>
                    {badge}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Contribution & Support Section */}
      <section className="py-24 px-4 bg-dark-bg border-t border-white/10">
        <div className="container mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-black mb-6">
              <span className="bg-gradient-to-r from-primary-400 to-purple-400 bg-clip-text text-transparent">
                Support This Project
              </span>
            </h2>
            <p className="text-xl text-neutral-400">
              Interested in deploying this project or want to contribute? We welcome your support!
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-5xl mx-auto">
            {/* GitHub Contribution Card */}
            <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-3xl p-10 hover:border-primary-500/50 transition-all duration-300">
              <div className="w-16 h-16 bg-gradient-to-br from-primary-500/20 to-purple-500/20 rounded-2xl flex items-center justify-center mb-6 mx-auto">
                <FontAwesomeIcon icon={faCode} className="text-4xl text-primary-400" />
              </div>
              
              <h3 className="text-2xl font-bold text-white mb-4 text-center">
                Deploy & Contribute
              </h3>
              
              <p className="text-neutral-400 mb-8 text-center leading-relaxed">
                Want to deploy this project for yourself? Interested in contributing to make it even better? Check out our GitHub repository and join the development!
              </p>

              <a 
                href="https://github.com/RAYDENFLY/AuthenticAlgo/tree/main"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-center gap-3 bg-neutral-800 hover:bg-neutral-700 text-white font-bold px-8 py-4 rounded-xl transition-all duration-300 group"
              >
                <FontAwesomeIcon icon={faGithub} className="text-2xl" />
                View on GitHub
                <FontAwesomeIcon icon={faArrowUp} className="rotate-45 group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform" />
              </a>
            </div>

            {/* Donation Card */}
            <div className="bg-gradient-to-br from-[#F3BA2F]/10 to-[#F3BA2F]/5 backdrop-blur-sm border border-[#F3BA2F]/30 rounded-3xl p-10 hover:border-[#F3BA2F]/50 transition-all duration-300">
              <div className="w-16 h-16 bg-[#F3BA2F]/20 rounded-2xl flex items-center justify-center mb-6 mx-auto">
                <FontAwesomeIcon icon={faHeart} className="text-4xl text-[#F3BA2F]" />
              </div>
              
              <h3 className="text-2xl font-bold text-white mb-4 text-center">
                Support Lead Developer
              </h3>
              
              <p className="text-neutral-400 mb-6 text-center leading-relaxed">
                Appreciate the work? Support the lead developer with a donation via Binance Pay
              </p>

              {/* QR Code */}
              <div className="bg-white rounded-2xl p-6 mb-6 mx-auto w-fit">
                <img 
                  src="https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=https://app.binance.com/qr/dplk12ffcd43b58f413a97770e65935450c7"
                  alt="Binance Donation QR Code"
                  className="w-48 h-48 mx-auto"
                />
              </div>

              <div className="text-center">
                <p className="text-sm text-neutral-500 mb-4">
                  Scan with Binance App
                </p>
                <a 
                  href="https://app.binance.com/qr/dplk12ffcd43b58f413a97770e65935450c7"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-3 bg-[#F3BA2F] hover:bg-[#F3BA2F]/90 text-neutral-900 font-bold px-8 py-4 rounded-xl transition-all duration-300"
                >
                  <FontAwesomeIcon icon={faGift} />
                  Open in Binance
                </a>
              </div>
            </div>
          </div>

          {/* Additional Info */}
          <div className="mt-12 text-center">
            <p className="text-neutral-500 text-sm">
              Your support helps us maintain and improve this project for the entire community
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}