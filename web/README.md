# AuthenticAlgo Pro - Frontend Dashboard

Clean, professional trading dashboard built with Next.js, TypeScript, and Tailwind CSS.

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS (no vanilla CSS)
- **Animations**: Anime.js + Framer Motion
- **Icons**: Font Awesome 6
- **Charts**: Lightweight Charts + Recharts
- **Data Fetching**: SWR + Axios
- **Real-time**: WebSocket

## Installation

```bash
cd web

# Install dependencies
npm install

# Or with yarn
yarn install
```

## Development

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

The app will run on **http://localhost:3000**

## Project Structure

```
web/
├── src/
│   ├── app/
│   │   ├── layout.tsx          # Root layout with Navbar/Footer
│   │   ├── page.tsx             # Homepage
│   │   ├── globals.css          # Tailwind imports
│   │   ├── dashboard/           # Trading dashboard
│   │   ├── arena/               # Strategy competition
│   │   └── ml/                  # ML models page
│   ├── components/
│   │   ├── Navbar.tsx           # Navigation bar
│   │   ├── Footer.tsx           # Footer
│   │   ├── StatCard.tsx         # Stat display card
│   │   ├── EquityChart.tsx      # Equity curve chart
│   │   ├── PositionCard.tsx     # Position display
│   │   ├── SignalCard.tsx       # Trading signal
│   │   └── ...
│   ├── lib/
│   │   ├── api.ts               # API client (Axios)
│   │   ├── hooks.ts             # Custom React hooks (SWR)
│   │   └── utils.ts             # Utility functions
│   └── styles/
│       └── globals.css          # Global Tailwind styles
├── public/                      # Static assets
├── tailwind.config.js           # Tailwind configuration
├── tsconfig.json                # TypeScript configuration
├── next.config.js               # Next.js configuration
└── package.json                 # Dependencies
```

## Features

### Homepage
- Hero section with live stats
- Feature highlights
- Strategy comparison widget
- Performance metrics
- Call-to-action sections

### Dashboard (`/dashboard`)
- Real-time PnL and balance
- Live equity curve chart
- Current open positions
- Recent trade history
- Portfolio breakdown
- WebSocket live updates

### Arena (`/arena`)
- Strategy competition view
- 3-way comparison (TA vs ML vs Hybrid)
- Live leaderboard
- Trade-by-trade analysis
- Start new competition

### ML Models (`/ml`)
- Model performance metrics
- Feature importance charts
- Confusion matrix
- Prediction analytics
- Backtest results

## Environment Variables

Create `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

## Design Principles

### Colors
- **Dark Theme**: Professional trader aesthetic
- **Primary**: Purple/Blue gradient (#6366f1)
- **Bull**: Green (#10b981)
- **Bear**: Red (#ef4444)

### Typography
- **Font**: Inter (sans-serif)
- **Mono**: JetBrains Mono (numbers/prices)

### Components
- **Card**: Glass-morphism effect
- **Buttons**: Gradient hover with shadows
- **Badges**: Semantic colors (bull/bear/neutral)
- **Inputs**: Dark theme with focus rings

### Animations
- **Page Load**: Fade in + slide up
- **Cards**: Stagger animation
- **Numbers**: Count-up effect
- **Charts**: Smooth transitions

## API Integration

### SWR for Data Fetching
```typescript
// Auto-refreshing data
const { data, error, isLoading } = usePublicStats();
```

### WebSocket for Real-time
```typescript
const { data, isConnected } = useWebSocket('ws://localhost:8000/ws/live');
```

### API Endpoints
All API calls are in `src/lib/api.ts`:
- `publicAPI.*` - Public endpoints
- `tradingAPI.*` - Trading data
- `mlAPI.*` - ML models
- `arenaAPI.*` - Competition
- `adminAPI.*` - Admin panel

## Development Tips

### Adding New Page
1. Create file in `src/app/your-page/page.tsx`
2. Use `'use client'` if you need client-side hooks
3. Import components from `@/components/`

### Adding New Component
1. Create in `src/components/YourComponent.tsx`
2. Use TypeScript interfaces for props
3. Style with Tailwind classes
4. Add animations with Anime.js or Framer Motion

### Styling Guidelines
- **Always use Tailwind** - No vanilla CSS
- **Use custom classes** from `globals.css`:
  - `.card`, `.card-hover`
  - `.btn-primary`, `.btn-secondary`
  - `.badge-bull`, `.badge-bear`
  - `.price-up`, `.price-down`
- **Responsive**: Mobile-first (use `md:`, `lg:` breakpoints)

## Testing

```bash
# Run linter
npm run lint

# Type checking
npx tsc --noEmit
```

## Building for Production

```bash
# Build optimized bundle
npm run build

# Test production build locally
npm start
```

## Deployment

### Vercel (Recommended)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

### Docker
```dockerfile
FROM node:20-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]
```

## Performance

- **Lighthouse Score**: 95+ (aim for 100)
- **First Paint**: < 1s
- **Interactive**: < 2s
- **Bundle Size**: < 200KB (gzipped)

## Browser Support

- Chrome/Edge: Latest 2 versions
- Firefox: Latest 2 versions
- Safari: Latest 2 versions
- Mobile: iOS Safari, Chrome Android

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 3000
npx kill-port 3000

# Or use different port
npm run dev -- -p 3001
```

### Build Errors
```bash
# Clean cache
rm -rf .next node_modules
npm install
npm run build
```

### API Connection Issues
- Check API is running on http://localhost:8000
- Verify `NEXT_PUBLIC_API_URL` in `.env.local`
- Check browser console for CORS errors

## Contributing

1. Always use TypeScript
2. Follow existing code style
3. Add comments for complex logic
4. Test responsive design (mobile/tablet/desktop)
5. Ensure animations are smooth (60fps)

## Support

- API Docs: http://localhost:8000/docs
- Issues: GitHub Issues
- Email: support@authenticalgo.com
