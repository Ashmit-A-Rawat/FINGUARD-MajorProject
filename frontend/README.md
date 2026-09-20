# FIN-GUARD review UI

React + Vite + TypeScript + Tailwind. Talks to the API under `/api` (proxied to `localhost:8000` in development).

    npm install
    npm run dev        # http://localhost:5173
    npm run build      # typecheck + production build into dist/
    npm test           # component and logic tests (vitest)

Start the API first (`make api`) and create a user (`FINGUARD_PASSWORD='...' python scripts/create_user.py alice analyst`).
The UI is advisory and shows SYNTHETIC data only. See `docs/architecture/system-architecture.md` for what each panel shows and why.
