// Chimera Service Worker for Performance Optimization
const CACHE_NAME = 'chimera-v1';
const STATIC_CACHE = 'chimera-static-v1';
const DYNAMIC_CACHE = 'chimera-dynamic-v1';

// Assets to cache immediately
const STATIC_ASSETS = [
  '/',
  '/manifest.json',
  '/favicon.ico',
  '/_next/static/css/app/layout.css',
  '/_next/static/chunks/webpack.js',
  '/_next/static/chunks/main.js',
  '/_next/static/chunks/framework.js',
  '/_next/static/chunks/pages/_app.js',
  '/fonts/inter-latin.woff2',
];

// Cache strategies
const CACHE_STRATEGIES = {
  // Cache first for static assets
  CACHE_FIRST: 'cache-first',
  // Network first for API calls
  NETWORK_FIRST: 'network-first',
  // Stale while revalidate for pages
  STALE_WHILE_REVALIDATE: 'stale-while-revalidate',
  // Network only for critical API calls
  NETWORK_ONLY: 'network-only',
};

// URL patterns and their cache strategies
const CACHE_RULES = [
  { pattern: /^\/api\/analytics/, strategy: CACHE_STRATEGIES.NETWORK_ONLY },
  { pattern: /^\/api\/.*/, strategy: CACHE_STRATEGIES.NETWORK_FIRST },
  { pattern: /\/_next\/static\/.*/, strategy: CACHE_STRATEGIES.CACHE_FIRST },
  { pattern: /\.(?:js|css|woff2|png|jpg|jpeg|svg|webp|avif)$/, strategy: CACHE_STRATEGIES.CACHE_FIRST },
  { pattern: /^\/dashboard\/.*/, strategy: CACHE_STRATEGIES.STALE_WHILE_REVALIDATE },
  { pattern: /.*/, strategy: CACHE_STRATEGIES.NETWORK_FIRST },
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('SW: Precaching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => {
        return self.skipWaiting();
      })
  );
});

// Activate event - cleanup old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (cacheName !== STATIC_CACHE && cacheName !== DYNAMIC_CACHE) {
              console.log('SW: Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        return self.clients.claim();
      })
  );
});

// Fetch event - implement caching strategies
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }

  // Skip chrome-extension and other non-http requests
  if (!request.url.startsWith('http')) {
    return;
  }

  // Find matching cache strategy
  const rule = CACHE_RULES.find(rule => rule.pattern.test(url.pathname));
  const strategy = rule ? rule.strategy : CACHE_STRATEGIES.NETWORK_FIRST;

  event.respondWith(handleRequest(request, strategy));
});

// Handle requests based on strategy
async function handleRequest(request, strategy) {
  const url = new URL(request.url);

  switch (strategy) {
    case CACHE_STRATEGIES.CACHE_FIRST:
      return cacheFirst(request);

    case CACHE_STRATEGIES.NETWORK_FIRST:
      return networkFirst(request);

    case CACHE_STRATEGIES.STALE_WHILE_REVALIDATE:
      return staleWhileRevalidate(request);

    case CACHE_STRATEGIES.NETWORK_ONLY:
      return fetch(request);

    default:
      return networkFirst(request);
  }
}

// Cache first strategy - for static assets
async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) {
    return cached;
  }

  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    console.error('SW: Cache first failed:', error);
    return new Response('Offline', { status: 503 });
  }
}

// Network first strategy - for API calls
async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    console.log('SW: Network failed, trying cache:', error);
    const cached = await caches.match(request);
    if (cached) {
      return cached;
    }
    return new Response('Offline', {
      status: 503,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: 'Network unavailable' })
    });
  }
}

// Stale while revalidate - for pages
async function staleWhileRevalidate(request) {
  const cached = await caches.match(request);

  // Start fetch in background
  const fetchPromise = fetch(request)
    .then((response) => {
      if (response.ok) {
        const cache = caches.open(DYNAMIC_CACHE);
        cache.then(c => c.put(request, response.clone()));
      }
      return response;
    })
    .catch(error => {
      console.error('SW: Background fetch failed:', error);
    });

  // Return cached version immediately if available
  if (cached) {
    return cached;
  }

  // If no cache, wait for network
  return fetchPromise;
}

// Handle background sync for analytics
self.addEventListener('sync', (event) => {
  if (event.tag === 'analytics-sync') {
    event.waitUntil(syncAnalytics());
  }
});

// Background sync for analytics data
async function syncAnalytics() {
  try {
    // Get pending analytics data from IndexedDB
    const pendingData = await getPendingAnalytics();

    if (pendingData.length > 0) {
      for (const data of pendingData) {
        try {
          await fetch('/api/analytics/web-vitals', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
          });
          // Remove from pending queue on success
          await removePendingAnalytics(data.id);
        } catch (error) {
          console.error('SW: Analytics sync failed:', error);
        }
      }
    }
  } catch (error) {
    console.error('SW: Sync analytics failed:', error);
  }
}

// IndexedDB helpers for analytics queue
async function getPendingAnalytics() {
  return new Promise((resolve) => {
    const request = indexedDB.open('chimera-analytics', 1);

    request.onsuccess = () => {
      const db = request.result;
      const transaction = db.transaction(['pending'], 'readonly');
      const store = transaction.objectStore('pending');
      const getAllRequest = store.getAll();

      getAllRequest.onsuccess = () => {
        resolve(getAllRequest.result || []);
      };
    };

    request.onerror = () => resolve([]);
  });
}

async function removePendingAnalytics(id) {
  return new Promise((resolve) => {
    const request = indexedDB.open('chimera-analytics', 1);

    request.onsuccess = () => {
      const db = request.result;
      const transaction = db.transaction(['pending'], 'readwrite');
      const store = transaction.objectStore('pending');
      store.delete(id);

      transaction.oncomplete = () => resolve();
    };

    request.onerror = () => resolve();
  });
}

// Performance monitoring
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'PERFORMANCE_MARK') {
    // Log performance marks for debugging
    console.log('SW: Performance mark:', event.data);
  }
});

// Cache cleanup on quota exceeded
self.addEventListener('quotaexceeded', (event) => {
  console.warn('SW: Quota exceeded, cleaning up caches');
  caches.keys().then((cacheNames) => {
    // Delete oldest caches first
    const oldCaches = cacheNames.filter(name =>
      name.startsWith('chimera-dynamic') || name.startsWith('chimera-v')
    ).slice(0, -2); // Keep only the latest 2

    return Promise.all(oldCaches.map(cache => caches.delete(cache)));
  });
});