import type { Metadata } from 'next';
import './globals.css';
import { Inter } from 'next/font/google';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'AuthenticAlgo Pro - AI Trading Bot',
  description: 'Transform $5 into $100 with AI-powered algorithmic trading. 96% accuracy, professional risk management.',
  keywords: 'trading bot, AI trading, algorithmic trading, crypto trading, forex trading',
  authors: [{ name: 'AuthenticAlgo Team' }],
  openGraph: {
    title: 'AuthenticAlgo Pro - AI Trading Bot',
    description: 'AI-powered trading bot with 96% accuracy',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-dark-bg text-white`}>
        <Navbar />
        <main className="min-h-screen pt-20 pb-8">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  );
}
