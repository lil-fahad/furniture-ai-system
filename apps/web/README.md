# FurnitureAI Web

Next.js web studio for the FurnitureAI v2 platform.

## Local

```bash
cp .env.example .env.local
npm install
npm run dev
```

`FURNITURE_API_URL` points to the FastAPI service. `FURNITURE_SERVICE_KEY` is server-only and is never exposed to browser code.

## Vercel

Create/link a Vercel project with **Root Directory** set to `apps/web`, Node.js `24.x`, then configure the two server environment variables. Preview deployments should target a non-production FurnitureAI API until the v2 backend passes production gates.
