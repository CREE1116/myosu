import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { defineConfig, type Plugin } from 'vite';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ALLOWED_MONITOR_FILES = new Set(['live_training.json', 'rl_replays.json']);

/** Live training checkpoints when present; otherwise the released model committed in models/. */
function checkpointDir(): string {
  const live = path.resolve(__dirname, 'arch-bench/checkpoints');
  return fs.existsSync(path.join(live, 'manifest.json')) ? live : path.resolve(__dirname, 'models');
}

function resolveLatestMonitorFile(fileName: string): string | null {
  const candidates = [
    path.resolve(__dirname, 'arch-bench/web_data', fileName),
    path.resolve(__dirname, 'public', fileName),
  ];

  let bestPath: string | null = null;
  let bestMtime = -1;
  for (const candidate of candidates) {
    if (!fs.existsSync(candidate)) continue;
    const stat = fs.statSync(candidate);
    if (stat.mtimeMs > bestMtime) {
      bestMtime = stat.mtimeMs;
      bestPath = candidate;
    }
  }
  return bestPath;
}

function attachMonitorMiddleware(middlewares: {
  use: (
    fn: (
      req: IncomingMessage,
      res: ServerResponse,
      next: (err?: unknown) => void,
    ) => void,
  ) => void;
}) {
  middlewares.use((req, res, next) => {
    const rawUrl = req.url ?? '/';
    const [pathname, search] = rawUrl.split('?');

    if (pathname === '/monitor' || pathname === '/monitor/') {
      req.url = search ? `/monitor.html?${search}` : '/monitor.html';
      next();
      return;
    }

    if (pathname === '/checkpoints/manifest.json' || /^\/checkpoints\/gen_\d{3}\.bin$/.test(pathname)) {
      const fileName = pathname.slice('/checkpoints/'.length);
      const filePath = path.join(checkpointDir(), fileName);
      if (!fs.existsSync(filePath)) {
        res.statusCode = 404;
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        res.end('{}');
        return;
      }

      const payload = fs.readFileSync(filePath);
      res.statusCode = 200;
      res.setHeader(
        'Content-Type',
        fileName.endsWith('.json')
          ? 'application/json; charset=utf-8'
          : 'application/octet-stream',
      );
      res.setHeader('Cache-Control', 'no-store');
      res.end(payload);
      return;
    }

    if (pathname.startsWith('/monitor-data/')) {
      const fileName = pathname.slice('/monitor-data/'.length);
      if (!ALLOWED_MONITOR_FILES.has(fileName)) {
        res.statusCode = 404;
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        res.end('{}');
        return;
      }

      const filePath = resolveLatestMonitorFile(fileName);
      if (!filePath) {
        res.statusCode = 404;
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        res.end('{}');
        return;
      }

      const payload = fs.readFileSync(filePath);
      res.statusCode = 200;
      res.setHeader('Content-Type', 'application/json; charset=utf-8');
      res.setHeader('Cache-Control', 'no-store');
      res.end(payload);
      return;
    }

    next();
  });
}

function monitorRoutesPlugin(): Plugin {
  return {
    name: 'gomoku-monitor-routes',
    configureServer(server) {
      attachMonitorMiddleware(server.middlewares);
    },
    configurePreviewServer(server) {
      attachMonitorMiddleware(server.middlewares);
    },
  };
}

function bundleChampionCheckpointPlugin(): Plugin {
  return {
    name: 'gomoku-bundle-champion-checkpoint',
    apply: 'build',
    closeBundle() {
      const srcDir = checkpointDir();
      const manifestPath = path.join(srcDir, 'manifest.json');
      if (!fs.existsSync(manifestPath)) return;

      const raw = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as {
        modelName?: string;
        params?: number;
        championGen?: number | null;
        updatedAt?: string;
        checkpoints?: Array<{
          id: string;
          gen: number;
          binFile: string;
          elo?: number;
          crit?: number;
          vsRbRecord?: string;
          vsRbWr?: number;
          isChampion?: boolean;
        }>;
      };
      const list = raw.checkpoints ?? [];
      if (list.length === 0) return;

      const forcedGen = process.env.GOMOKU_DEPLOY_GEN
        ? Number(process.env.GOMOKU_DEPLOY_GEN)
        : null;
      const champ =
        (forcedGen != null && Number.isFinite(forcedGen)
          ? list.find((c) => c.gen === forcedGen)
          : null) ??
        list.find(
          (c) =>
            Boolean(c.isChampion) ||
            (raw.championGen != null && c.gen === raw.championGen),
        ) ??
        [...list].sort((a, b) => b.gen - a.gen)[0];

      if (!champ?.binFile) return;
      const srcBin = path.join(srcDir, champ.binFile);
      if (!fs.existsSync(srcBin)) return;

      const outDir = path.resolve(__dirname, 'dist/checkpoints');
      fs.mkdirSync(outDir, { recursive: true });
      fs.copyFileSync(srcBin, path.join(outDir, champ.binFile));

      const slimManifest = {
        modelName: raw.modelName ?? 'D4OverlapPattern + Rust MCTS',
        params: raw.params ?? 61472,
        championGen: champ.gen,
        updatedAt: raw.updatedAt ?? new Date().toISOString(),
        checkpoints: [
          {
            id: champ.id,
            gen: champ.gen,
            binFile: champ.binFile,
            elo: champ.elo,
            crit: champ.crit,
            vsRbRecord: champ.vsRbRecord,
            vsRbWr: champ.vsRbWr,
            isChampion: true,
          },
        ],
      };
      fs.writeFileSync(
        path.join(outDir, 'manifest.json'),
        JSON.stringify(slimManifest, null, 2),
        'utf-8',
      );
    },
  };
}

export default defineConfig({
  base: './',
  // Only the PWA files (icons, manifest, service worker) ship as-is; public/ holds work-in-progress art.
  publicDir: 'static',
  plugins: [monitorRoutesPlugin(), bundleChampionCheckpointPlugin()],
});
