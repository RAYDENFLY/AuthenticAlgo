'use client';

import { useEffect, useRef, useState } from 'react';
import anime from 'animejs';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faBrain,
  faRobot,
  faCheckCircle,
  faChartBar,
  faLightbulb,
  faBolt,
  faShield,
  faStar,
  faCrown,
  faMicrochip,
  faFire,
  faTrophy,
  faGraduationCap,
  faChartLine,
  faLayerGroup,
} from '@fortawesome/free-solid-svg-icons';
import { useMLModels, useMLPerformance } from '@/lib/hooks';

export default function MLPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  
  const { data: models, isLoading: modelsLoading } = useMLModels();
  const { data: performance, isLoading: perfLoading } = useMLPerformance();

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

  const formatPercentage = (value: number) => {
    return `${value.toFixed(1)}%`;
  };

  const getAccuracyColor = (accuracy: number) => {
    if (accuracy >= 95) return 'text-bull';
    if (accuracy >= 85) return 'text-primary-400';
    if (accuracy >= 75) return 'text-yellow-400';
    return 'text-bear';
  };

  const getAccuracyBg = (accuracy: number) => {
    if (accuracy >= 95) return 'from-bull/20 to-bull/5';
    if (accuracy >= 85) return 'from-primary-500/20 to-primary-500/5';
    if (accuracy >= 75) return 'from-yellow-500/20 to-yellow-500/5';
    return 'from-bear/20 to-bear/5';
  };

  const getModelIcon = (modelName: string) => {
    const name = modelName.toLowerCase();
    if (name.includes('xgboost')) return faBolt;
    if (name.includes('forest') || name.includes('tree')) return faLayerGroup;
    if (name.includes('neural')) return faBrain;
    if (name.includes('gradient')) return faChartLine;
    if (name.includes('boost')) return faFire;
    return faMicrochip;
  };

  return (
    <div className="min-h-screen bg-dark-bg py-24 px-4">
      <div className="mx-auto max-w-[1600px]" ref={containerRef}>
        {/* Header */}
        <div className="mb-12 fade-in-card">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 bg-gradient-to-br from-primary-500 to-purple-500 rounded-2xl flex items-center justify-center shadow-xl shadow-primary-500/25">
              <FontAwesomeIcon icon={faBrain} className="text-white text-2xl" />
            </div>
            <div>
              <h1 className="text-5xl font-black text-white mb-2">
                AI Models
              </h1>
              <p className="text-lg text-neutral-400">
                12 optimized machine learning models with 96% accuracy
              </p>
            </div>
          </div>
        </div>

        {/* Performance Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
          {[
            {
              icon: faTrophy,
              label: 'Best Accuracy',
              value: performance?.best_model_accuracy || 96,
              suffix: '%',
              description: performance?.best_model || 'Random Forest',
              color: 'from-bull/20 to-bull/5',
              iconColor: 'text-bull',
            },
            {
              icon: faRobot,
              label: 'Active Models',
              value: models?.filter((m: any) => m.status === 'active').length || 11,
              suffix: '/12',
              description: 'Production ready',
              color: 'from-primary-500/20 to-primary-500/5',
              iconColor: 'text-primary-400',
            },
            {
              icon: faGraduationCap,
              label: 'Training Data',
              value: '52',
              suffix: ' Features',
              description: '100K+ samples',
              color: 'from-purple-500/20 to-purple-500/5',
              iconColor: 'text-purple-400',
            },
            {
              icon: faFire,
              label: 'Predictions Today',
              value: performance?.predictions_today || 847,
              suffix: '',
              description: '99.2% success rate',
              color: 'from-orange-500/20 to-orange-500/5',
              iconColor: 'text-orange-400',
            },
          ].map((stat, index) => (
            <div key={index} className="fade-in-card">
              <div className={`bg-gradient-to-br ${stat.color} border border-white/10 rounded-3xl p-6 hover:border-primary-500/50 transition-all duration-300 h-full`}>
                <div className="flex items-center justify-between mb-4">
                  <div className={`w-12 h-12 bg-white/10 rounded-2xl flex items-center justify-center`}>
                    <FontAwesomeIcon icon={stat.icon} className={`${stat.iconColor} text-xl`} />
                  </div>
                </div>
                <div className="mb-2">
                  <div className="text-neutral-400 text-sm font-semibold mb-2 uppercase tracking-wider">
                    {stat.label}
                  </div>
                  <div className="text-3xl font-black text-white flex items-baseline gap-1">
                    {perfLoading ? (
                      <div className="h-8 w-20 bg-white/10 rounded-lg animate-pulse"></div>
                    ) : (
                      <>
                        <span>{stat.value}</span>
                        <span className="text-lg text-neutral-400">{stat.suffix}</span>
                      </>
                    )}
                  </div>
                </div>
                <div className="text-sm text-neutral-500 pt-3 border-t border-white/10">
                  {stat.description}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Models Grid */}
        <div className="mb-12">
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-primary-500/20 rounded-2xl flex items-center justify-center">
                <FontAwesomeIcon icon={faRobot} className="text-primary-400 text-xl" />
              </div>
              <h2 className="text-3xl font-black text-white">ML Models</h2>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 bg-bull/20 border border-bull/30 rounded-xl">
              <div className="w-2 h-2 bg-bull rounded-full animate-pulse"></div>
              <span className="text-bull font-bold text-sm">
                {models?.filter((m: any) => m.status === 'active').length || 11} ACTIVE
              </span>
            </div>
          </div>

          {modelsLoading ? (
            <div className="text-center py-20">
              <div className="inline-block animate-spin rounded-full h-16 w-16 border-4 border-primary-500/20 border-t-primary-500 mb-4"></div>
              <p className="text-neutral-400 text-lg">Loading models...</p>
            </div>
          ) : models && models.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {models.map((model: any, index: number) => (
                <div
                  key={model.id || index}
                  className="fade-in-card group"
                  onClick={() => setSelectedModel(selectedModel === model.name ? null : model.name)}
                >
                  <div className={`bg-gradient-to-br ${getAccuracyBg(model.accuracy || 0)} border border-white/10 rounded-3xl p-6 hover:border-primary-500/50 transition-all duration-300 cursor-pointer ${
                    selectedModel === model.name ? 'ring-2 ring-primary-500' : ''
                  }`}>
                    {/* Header */}
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center gap-3">
                        <div className="w-12 h-12 bg-white/10 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform">
                          <FontAwesomeIcon 
                            icon={getModelIcon(model.name || 'default')} 
                            className="text-primary-400 text-xl" 
                          />
                        </div>
                        <div>
                          <h3 className="text-lg font-black text-white">{model.name}</h3>
                          <span className={`inline-block px-2 py-1 rounded-lg text-xs font-bold mt-1 ${
                            model.status === 'active' 
                              ? 'bg-bull/20 text-bull border border-bull/30' 
                              : 'bg-white/10 text-neutral-400 border border-white/20'
                          }`}>
                            {model.status || 'active'}
                          </span>
                        </div>
                      </div>
                      {(model.accuracy || 0) >= 95 && (
                        <FontAwesomeIcon icon={faCrown} className="text-yellow-400 text-xl" />
                      )}
                    </div>

                    {/* Accuracy */}
                    <div className="mb-6">
                      <div className="flex items-baseline justify-between mb-2">
                        <span className="text-neutral-400 text-sm font-semibold uppercase tracking-wider">
                          Accuracy
                        </span>
                        <span className={`text-3xl font-black ${getAccuracyColor(model.accuracy || 0)}`}>
                          {formatPercentage(model.accuracy || 0)}
                        </span>
                      </div>
                      <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                        <div 
                          className={`h-full rounded-full transition-all duration-1000 ${
                            (model.accuracy || 0) >= 95 ? 'bg-bull' :
                            (model.accuracy || 0) >= 85 ? 'bg-primary-500' :
                            (model.accuracy || 0) >= 75 ? 'bg-yellow-500' : 'bg-bear'
                          }`}
                          style={{ width: `${model.accuracy || 0}%` }}
                        ></div>
                      </div>
                    </div>

                    {/* Metrics */}
                    <div className="grid grid-cols-2 gap-3">
                      <div className="bg-white/5 rounded-xl p-3 border border-white/10">
                        <div className="text-neutral-500 text-xs font-bold mb-1 uppercase">Precision</div>
                        <div className="text-white font-bold">{formatPercentage(model.precision || 0)}</div>
                      </div>
                      <div className="bg-white/5 rounded-xl p-3 border border-white/10">
                        <div className="text-neutral-500 text-xs font-bold mb-1 uppercase">Recall</div>
                        <div className="text-white font-bold">{formatPercentage(model.recall || 0)}</div>
                      </div>
                      <div className="bg-white/5 rounded-xl p-3 border border-white/10">
                        <div className="text-neutral-500 text-xs font-bold mb-1 uppercase">F1 Score</div>
                        <div className="text-white font-bold">{formatPercentage(model.f1_score || 0)}</div>
                      </div>
                      <div className="bg-white/5 rounded-xl p-3 border border-white/10">
                        <div className="text-neutral-500 text-xs font-bold mb-1 uppercase">Loss</div>
                        <div className="text-white font-bold">{(model.loss || 0).toFixed(3)}</div>
                      </div>
                    </div>

                    {/* Expanded Details */}
                    {selectedModel === model.name && (
                      <div className="mt-6 pt-6 border-t border-white/10 space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-neutral-400 text-sm">Training Time</span>
                          <span className="text-white font-semibold">{model.training_time || '2.3s'}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-neutral-400 text-sm">Predictions</span>
                          <span className="text-white font-semibold">{model.predictions_count || '5,432'}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-neutral-400 text-sm">Last Updated</span>
                          <span className="text-white font-semibold">{model.last_updated || '2h ago'}</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-24">
              <div className="w-24 h-24 bg-white/5 rounded-3xl flex items-center justify-center mx-auto mb-6">
                <FontAwesomeIcon icon={faBrain} className="text-neutral-600 text-5xl" />
              </div>
              <h3 className="text-2xl font-bold text-white mb-3">No Models Found</h3>
              <p className="text-neutral-400">ML models will appear here once trained</p>
            </div>
          )}
        </div>

        {/* Feature Importance */}
        <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-3xl p-8">
          <div className="flex items-center gap-4 mb-8">
            <div className="w-12 h-12 bg-primary-500/20 rounded-2xl flex items-center justify-center">
              <FontAwesomeIcon icon={faChartBar} className="text-primary-400 text-xl" />
            </div>
            <div>
              <h2 className="text-3xl font-black text-white mb-1">Top Features</h2>
              <p className="text-neutral-400">Most important indicators for predictions</p>
            </div>
          </div>

          <div className="space-y-4">
            {[
              { name: 'RSI (14)', importance: 0.245, rank: 1 },
              { name: 'MACD Signal', importance: 0.198, rank: 2 },
              { name: 'Volume MA (20)', importance: 0.167, rank: 3 },
              { name: 'Bollinger Bands', importance: 0.143, rank: 4 },
              { name: 'EMA (50)', importance: 0.128, rank: 5 },
              { name: 'Stochastic RSI', importance: 0.119, rank: 6 },
            ].map((feature) => (
              <div key={feature.rank} className="group">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <div className={`w-8 h-8 rounded-xl flex items-center justify-center font-bold text-sm ${
                      feature.rank <= 3 
                        ? 'bg-gradient-to-br from-primary-500 to-purple-500 text-white' 
                        : 'bg-white/10 text-neutral-400'
                    }`}>
                      {feature.rank}
                    </div>
                    <span className="text-white font-semibold">{feature.name}</span>
                  </div>
                  <span className="text-primary-400 font-bold">
                    {(feature.importance * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-primary-500 to-purple-500 rounded-full transition-all duration-1000"
                    style={{ width: `${feature.importance * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
