'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faChartLine, 
  faTrophy, 
  faRobot, 
  faBars, 
  faTimes,
  faGauge,
  faChartBar,
  faBook,
  faFileAlt,
  faChevronDown
} from '@fortawesome/free-solid-svg-icons';
import { faDiscord } from '@fortawesome/free-brands-svg-icons';
import anime from 'animejs';

export default function Navbar() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isResourcesOpen, setIsResourcesOpen] = useState(false);
  const [livePnL, setLivePnL] = useState('+$45.23');
  const pathname = usePathname();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };

    // Simulate live PnL updates
    const pnlInterval = setInterval(() => {
      setLivePnL(`+$${(45.23 + Math.random() * 2).toFixed(2)}`);
    }, 5000);

    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
      clearInterval(pnlInterval);
    };
  }, []);

  useEffect(() => {
    // Animate nav items on mount
    anime({
      targets: '.nav-item',
      opacity: [0, 1],
      translateY: [-15, 0],
      delay: anime.stagger(100, { start: 300 }),
      duration: 800,
      easing: 'easeOutExpo',
    });

    // Animate live badge
    anime({
      targets: '.live-badge',
      scale: [0.8, 1],
      opacity: [0, 1],
      duration: 1000,
      easing: 'easeOutBack',
    });
  }, []);

  const navLinks = [
    { href: '/dashboard', label: 'Dashboard', icon: faGauge, badge: 'Live' },
    { href: '/arena', label: 'Arena', icon: faTrophy, badge: 'Hot' },
    { href: '/ml', label: 'AI Models', icon: faRobot, badge: '96%' },
    { href: '/analytics', label: 'Analytics', icon: faChartBar, badge: 'New' },
  ];

  const resourceLinks = [
    { href: '/reports', label: 'Performance Reports', icon: faFileAlt },
    { href: '/documentation', label: 'Documentation', icon: faBook },
  ];

  const isActive = (href: string) => pathname === href;

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        isScrolled 
          ? 'bg-white/5 backdrop-blur-xl border-b border-white/10 shadow-2xl shadow-primary-500/10' 
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-[1800px] mx-auto px-8 xl:px-12">
        <div className="flex items-center justify-between h-20">
          {/* Logo */}
          <Link 
            href="/" 
            className="flex items-center gap-3 nav-item group"
          >
            <div className="relative">
              <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-purple-500 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300 shadow-lg shadow-primary-500/25">
                <FontAwesomeIcon icon={faChartLine} className="text-white text-lg" />
              </div>
              {/* Animated pulse dot */}
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-bull rounded-full border-2 border-dark-bg animate-pulse"></div>
            </div>
            <div className="flex flex-col">
              <span className="text-2xl font-black bg-gradient-to-r from-primary-400 to-purple-400 bg-clip-text text-transparent leading-tight">
                AuthenticAlgo
              </span>
              <span className="text-xs text-bull font-bold tracking-wider">PRO</span>
            </div>
          </Link>

          {/* Desktop Nav */}
          <div className="hidden lg:flex items-center gap-2">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`nav-item group relative flex items-center gap-3 px-6 py-3 rounded-2xl transition-all duration-300 ${
                  isActive(link.href)
                    ? 'bg-primary-500/20 text-primary-400 shadow-lg shadow-primary-500/20'
                    : 'text-neutral-300 hover:text-white hover:bg-white/5'
                }`}
              >
                <FontAwesomeIcon 
                  icon={link.icon} 
                  className={`text-sm transition-transform duration-300 ${
                    isActive(link.href) ? 'scale-110' : 'group-hover:scale-110'
                  }`} 
                />
                <span className="font-semibold">{link.label}</span>
                
                {/* Badge */}
                <span className={`px-2 py-1 rounded-full text-xs font-bold ${
                  link.badge === 'Live' ? 'bg-bull/20 text-bull border border-bull/30' :
                  link.badge === 'Hot' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' :
                  link.badge === '96%' ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30' :
                  'bg-purple-500/20 text-purple-400 border border-purple-500/30'
                }`}>
                  {link.badge}
                </span>

                {/* Active indicator */}
                {isActive(link.href) && (
                  <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-1/2 h-0.5 bg-primary-400 rounded-full"></div>
                )}
              </Link>
            ))}

            {/* Resources Dropdown */}
            <div className="relative nav-item">
              <button
                onMouseEnter={() => setIsResourcesOpen(true)}
                onMouseLeave={() => setIsResourcesOpen(false)}
                className={`group relative flex items-center gap-2 px-6 py-3 rounded-2xl transition-all duration-300 ${
                  pathname === '/reports' || pathname === '/documentation'
                    ? 'bg-primary-500/20 text-primary-400 shadow-lg shadow-primary-500/20'
                    : 'text-neutral-300 hover:text-white hover:bg-white/5'
                }`}
              >
                <FontAwesomeIcon 
                  icon={faBook} 
                  className="text-sm transition-transform duration-300 group-hover:scale-110" 
                />
                <span className="font-semibold">Resources</span>
                <FontAwesomeIcon 
                  icon={faChevronDown} 
                  className={`text-xs transition-transform duration-300 ${isResourcesOpen ? 'rotate-180' : ''}`} 
                />
              </button>

              {/* Dropdown Menu */}
              {isResourcesOpen && (
                <div 
                  className="absolute top-full left-0 mt-2 w-64 bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl overflow-hidden"
                  onMouseEnter={() => setIsResourcesOpen(true)}
                  onMouseLeave={() => setIsResourcesOpen(false)}
                >
                  {resourceLinks.map((link) => (
                    <Link
                      key={link.href}
                      href={link.href}
                      className={`flex items-center gap-3 px-6 py-4 transition-all duration-300 ${
                        isActive(link.href)
                          ? 'bg-primary-500/20 text-primary-400'
                          : 'text-neutral-300 hover:text-white hover:bg-white/5'
                      }`}
                    >
                      <FontAwesomeIcon icon={link.icon} className="text-sm" />
                      <span className="font-semibold">{link.label}</span>
                    </Link>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Right Section - CTA & Live Data */}
          <div className="hidden lg:flex items-center gap-4">
            {/* Live PnL Badge */}
            <div className="nav-item flex items-center gap-3 bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl px-4 py-2">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-bull rounded-full animate-pulse"></div>
                <span className="text-sm font-semibold text-neutral-300">Live P&L</span>
              </div>
              <span className="text-bull font-bold text-sm">{livePnL}</span>
            </div>

            {/* Discord CTA Button */}
            <a 
              href="https://discord.gg/your-server-link"
              target="_blank"
              rel="noopener noreferrer"
              className="nav-item group relative bg-[#5865F2] hover:bg-[#4752C4] text-white font-semibold px-8 py-3 rounded-2xl transition-all duration-300 transform hover:scale-105 hover:shadow-2xl shadow-lg shadow-[#5865F2]/25"
            >
              <FontAwesomeIcon icon={faDiscord} className="mr-2" />
              Join Discord
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            </a>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="lg:hidden w-12 h-12 bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl flex items-center justify-center text-white hover:bg-white/10 transition-all duration-300"
          >
            <FontAwesomeIcon icon={isMobileMenuOpen ? faTimes : faBars} className="text-lg" />
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      {isMobileMenuOpen && (
        <div className="lg:hidden bg-white/5 backdrop-blur-xl border-t border-white/10">
          <div className="container mx-auto px-4 py-6">
            {/* Live Stats */}
            <div className="flex items-center justify-between mb-6 p-4 bg-white/5 rounded-2xl border border-white/10">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-bull rounded-full animate-pulse"></div>
                <span className="text-sm font-semibold text-neutral-300">Live Trading</span>
              </div>
              <span className="text-bull font-bold">{livePnL}</span>
            </div>

            {/* Mobile Nav Links */}
            <div className="space-y-2">
              {navLinks.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setIsMobileMenuOpen(false)}
                  className={`flex items-center justify-between p-4 rounded-2xl transition-all duration-300 ${
                    isActive(link.href)
                      ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30'
                      : 'text-neutral-300 hover:bg-white/5 border border-transparent'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <FontAwesomeIcon icon={link.icon} className="text-lg" />
                    <span className="font-semibold">{link.label}</span>
                  </div>
                  <span className={`px-2 py-1 rounded-full text-xs font-bold ${
                    link.badge === 'Live' ? 'bg-bull/20 text-bull' :
                    link.badge === 'Hot' ? 'bg-orange-500/20 text-orange-400' :
                    link.badge === '96%' ? 'bg-primary-500/20 text-primary-400' :
                    'bg-purple-500/20 text-purple-400'
                  }`}>
                    {link.badge}
                  </span>
                </Link>
              ))}

              {/* Resources Section in Mobile */}
              <div className="pt-4 border-t border-white/10">
                <div className="text-xs font-bold text-neutral-500 uppercase tracking-wider mb-2 px-4">
                  Resources
                </div>
                {resourceLinks.map((link) => (
                  <Link
                    key={link.href}
                    href={link.href}
                    onClick={() => setIsMobileMenuOpen(false)}
                    className={`flex items-center gap-3 p-4 rounded-2xl transition-all duration-300 ${
                      isActive(link.href)
                        ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30'
                        : 'text-neutral-300 hover:bg-white/5 border border-transparent'
                    }`}
                  >
                    <FontAwesomeIcon icon={link.icon} className="text-lg" />
                    <span className="font-semibold">{link.label}</span>
                  </Link>
                ))}
              </div>
            </div>

            {/* Mobile CTA */}
            <div className="mt-6">
              <a
                href="https://discord.gg/your-server-link"
                target="_blank"
                rel="noopener noreferrer"
                className="w-full bg-[#5865F2] hover:bg-[#4752C4] text-white font-semibold py-4 rounded-2xl text-center block transition-all duration-300 shadow-lg shadow-[#5865F2]/25"
              >
                <FontAwesomeIcon icon={faDiscord} className="mr-2" />
                Join Discord Community
              </a>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
}