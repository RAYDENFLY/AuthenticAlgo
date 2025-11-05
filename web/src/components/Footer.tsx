'use client';

import Link from 'next/link';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faGithub, 
  faTwitter, 
  faDiscord, 
  faTelegram,
  faYoutube
} from '@fortawesome/free-brands-svg-icons';
import { 
  faChartLine,
  faEnvelope,
  faRocket,
  faShield,
  faBolt
} from '@fortawesome/free-solid-svg-icons';

export default function Footer() {
  const currentYear = new Date().getFullYear();

  const features = [
    { icon: faRocket, text: 'AI-Powered Trading' },
    { icon: faShield, text: 'Smart Risk Management' },
    { icon: faBolt, text: 'Real-Time Execution' }
  ];

  return (
    <footer className="relative bg-gradient-to-b from-dark-bg to-primary-900/5 border-t border-white/10 mt-32">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary-500/5 via-transparent to-transparent" />
      
      <div className="relative z-10">
        {/* Feature Highlights */}
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 -mt-16 mb-16">
            {features.map((feature, index) => (
              <div 
                key={index}
                className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-6 text-center hover:bg-white/10 transition-all duration-300 group"
              >
                <div className="w-12 h-12 bg-primary-500/10 rounded-xl flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform duration-300">
                  <FontAwesomeIcon 
                    icon={feature.icon} 
                    className="text-xl text-primary-400" 
                  />
                </div>
                <h3 className="text-white font-semibold text-sm">
                  {feature.text}
                </h3>
              </div>
            ))}
          </div>
        </div>

        {/* Main Footer Content */}
        <div className="container mx-auto px-4 py-16">
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-12">
            {/* Brand Section */}
            <div className="lg:col-span-2">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-purple-500 rounded-xl flex items-center justify-center">
                  <FontAwesomeIcon icon={faChartLine} className="text-white text-lg" />
                </div>
                <div>
                  <span className="text-2xl font-black bg-gradient-to-r from-primary-400 to-purple-400 bg-clip-text text-transparent">
                    AuthenticAlgo
                  </span>
                  <div className="text-xs text-bull font-semibold mt-1">
                    PRO EDITION
                  </div>
                </div>
              </div>
              <p className="text-neutral-300 text-lg leading-relaxed mb-6 max-w-md">
                Transform your trading with institutional-grade AI algorithms. 
                Start with just $5 and experience 96% accuracy in live markets.
              </p>
              
              {/* Social Links */}
              <div className="flex gap-4 mb-6">
                {[
                  { icon: faGithub, href: 'https://github.com/authenticalgo', label: 'GitHub' },
                  { icon: faTwitter, href: 'https://twitter.com/authenticalgo', label: 'Twitter' },
                  { icon: faDiscord, href: 'https://discord.gg/authenticalgo', label: 'Discord' },
                  { icon: faTelegram, href: 'https://t.me/authenticalgo', label: 'Telegram' },
                  { icon: faYoutube, href: 'https://youtube.com/authenticalgo', label: 'YouTube' }
                ].map((social, index) => (
                  <a
                    key={index}
                    href={social.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="w-12 h-12 bg-white/5 hover:bg-primary-500 border border-white/10 hover:border-primary-500 rounded-xl flex items-center justify-center transition-all duration-300 group hover:scale-110"
                    aria-label={social.label}
                  >
                    <FontAwesomeIcon 
                      icon={social.icon} 
                      className="text-neutral-400 group-hover:text-white text-lg transition-colors" 
                    />
                  </a>
                ))}
              </div>

              {/* Email */}
              <a 
                href="mailto:support@authenticalgo.com"
                className="inline-flex items-center gap-3 text-neutral-300 hover:text-white transition-colors group"
              >
                <div className="w-10 h-10 bg-white/5 rounded-lg flex items-center justify-center group-hover:bg-primary-500 transition-colors">
                  <FontAwesomeIcon icon={faEnvelope} className="text-sm" />
                </div>
                <div>
                  <div className="text-sm text-neutral-400">Contact Support</div>
                  <div className="font-semibold">support@authenticalgo.com</div>
                </div>
              </a>
            </div>

            {/* Quick Links */}
            <div>
              <h3 className="text-white font-bold text-lg mb-6">Platform</h3>
              <ul className="space-y-3">
                {[
                  { name: 'Live Dashboard', href: '/dashboard' },
                  { name: 'Trading Arena', href: '/arena' },
                  { name: 'ML Models', href: '/ml' },
                  { name: 'Backtesting', href: '/backtest' },
                  { name: 'Paper Trading', href: '/paper' }
                ].map((link, index) => (
                  <li key={index}>
                    <Link 
                      href={link.href}
                      className="text-neutral-400 hover:text-primary-400 transition-colors flex items-center gap-2 group"
                    >
                      <div className="w-1.5 h-1.5 bg-primary-500 rounded-full opacity-0 group-hover:opacity-100 transition-opacity" />
                      {link.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>

            {/* Resources */}
            <div>
              <h3 className="text-white font-bold text-lg mb-6">Resources</h3>
              <ul className="space-y-3">
                {[
                  { name: 'Documentation', href: '/docs' },
                  { name: 'API Reference', href: '/api' },
                  { name: 'Tutorials', href: '/tutorials' },
                  { name: 'Blog', href: '/blog' },
                  { name: 'Changelog', href: '/changelog' }
                ].map((link, index) => (
                  <li key={index}>
                    <Link 
                      href={link.href}
                      className="text-neutral-400 hover:text-primary-400 transition-colors flex items-center gap-2 group"
                    >
                      <div className="w-1.5 h-1.5 bg-primary-500 rounded-full opacity-0 group-hover:opacity-100 transition-opacity" />
                      {link.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>

            {/* Legal */}
            <div>
              <h3 className="text-white font-bold text-lg mb-6">Company</h3>
              <ul className="space-y-3">
                {[
                  { name: 'About Us', href: '/about' },
                  { name: 'System Status', href: '/status' },
                  { name: 'Terms of Service', href: '/terms' },
                  { name: 'Privacy Policy', href: '/privacy' },
                  { name: 'Risk Disclosure', href: '/risk' }
                ].map((link, index) => (
                  <li key={index}>
                    <Link 
                      href={link.href}
                      className="text-neutral-400 hover:text-primary-400 transition-colors flex items-center gap-2 group"
                    >
                      <div className="w-1.5 h-1.5 bg-primary-500 rounded-full opacity-0 group-hover:opacity-100 transition-opacity" />
                      {link.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Bottom Bar */}
          <div className="border-t border-white/10 mt-16 pt-8">
            <div className="flex flex-col lg:flex-row justify-between items-center gap-6">
              {/* Copyright */}
              <div className="text-center lg:text-left">
                <p className="text-neutral-400 text-sm">
                  © {currentYear} AuthenticAlgo Pro. All rights reserved.
                </p>
                <p className="text-neutral-500 text-xs mt-1">
                  Made with ❤️ for the trading community
                </p>
              </div>

              {/* Badges */}
              <div className="flex flex-wrap gap-4 justify-center">
                <div className="flex items-center gap-2 text-xs text-neutral-400">
                  <div className="w-2 h-2 bg-bull rounded-full animate-pulse"></div>
                  System Operational
                </div>
                <div className="flex items-center gap-2 text-xs text-neutral-400">
                  <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                  SSL Secured
                </div>
                <div className="flex items-center gap-2 text-xs text-neutral-400">
                  <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                  99.9% Uptime
                </div>
              </div>

              {/* Legal Links */}
              <div className="flex gap-6 text-sm text-neutral-400">
                <Link href="/terms" className="hover:text-white transition-colors">
                  Terms
                </Link>
                <Link href="/privacy" className="hover:text-white transition-colors">
                  Privacy
                </Link>
                <Link href="/cookies" className="hover:text-white transition-colors">
                  Cookies
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}