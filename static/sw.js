// 묘수 service worker: works offline after the first visit, including the cat's model.
// Bump VERSION whenever this file's strategy changes; hashed /assets/ files version themselves.
const VERSION = 'myosu-v1';
const SHELL = ['./', './index.html', './manifest.webmanifest', './icons/icon-192.png', './icons/favicon-32.png'];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

async function networkFirst(request, cacheKey) {
  const cache = await caches.open(VERSION);
  try {
    const res = await fetch(request);
    if (res.ok) cache.put(cacheKey, res.clone());
    return res;
  } catch {
    const hit = await cache.match(cacheKey);
    if (hit) return hit;
    throw new Error('offline and not cached');
  }
}

async function cacheFirst(request) {
  const cache = await caches.open(VERSION);
  const hit = await cache.match(request);
  if (hit) return hit;
  const res = await fetch(request);
  if (res.ok) cache.put(request, res.clone());
  return res;
}

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  // Page loads: fresh when online, cached shell when offline.
  if (req.mode === 'navigate') {
    event.respondWith(networkFirst(req, './index.html'));
    return;
  }
  // The app fetches the model with a ?t= cache-buster; key the cache without it.
  if (url.origin === location.origin && url.pathname.includes('/checkpoints/')) {
    event.respondWith(networkFirst(req, url.origin + url.pathname));
    return;
  }
  // Content-hashed bundles, sprites, icons and the display font never change under the same URL.
  if (
    (url.origin === location.origin && (url.pathname.includes('/assets/') || url.pathname.includes('/icons/'))) ||
    url.hostname === 'fonts.gstatic.com'
  ) {
    event.respondWith(cacheFirst(req));
    return;
  }
  if (url.hostname === 'fonts.googleapis.com') {
    event.respondWith(networkFirst(req, req.url));
  }
});
