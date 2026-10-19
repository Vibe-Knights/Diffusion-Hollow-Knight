/**
 * Persistent model weight caching using the browser's Cache API.
 * Ensures large .onnx models (58MB) only download once and load instantly from disk on future visits.
 */

const CACHE_NAME = 'dhk-model-cache-v1';

export type ProgressCallback = (loaded: number, total: number, percentage: number) => void;

export async function fetchWithCache(
  url: string,
  onProgress?: ProgressCallback
): Promise<ArrayBuffer> {
  // 1. Try Cache API first
  if (typeof caches !== 'undefined') {
    try {
      const cache = await caches.open(CACHE_NAME);
      const match = await cache.match(url);
      if (match) {
        const buffer = await match.arrayBuffer();
        if (onProgress) {
          onProgress(buffer.byteLength, buffer.byteLength, 100);
        }
        return buffer;
      }
    } catch (e) {
      console.warn('[Cache] Could not read from Cache API, falling back to network fetch:', e);
    }
  }

  // 2. Fetch from network with progress tracking
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${response.status} ${response.statusText}`);
  }

  const contentLengthHeader = response.headers.get('content-length');
  const total = contentLengthHeader ? parseInt(contentLengthHeader, 10) : 0;

  let buffer: ArrayBuffer;

  if (response.body && total > 0) {
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let received = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        received += value.length;
        if (onProgress) {
          const percent = Math.min(100, Math.round((received / total) * 100));
          onProgress(received, total, percent);
        }
      }
    }

    const merged = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    buffer = merged.buffer;
  } else {
    buffer = await response.arrayBuffer();
    if (onProgress) {
      onProgress(buffer.byteLength, buffer.byteLength, 100);
    }
  }

  // 3. Store into Cache API for subsequent loads
  if (typeof caches !== 'undefined') {
    try {
      const cache = await caches.open(CACHE_NAME);
      const cacheResponse = new Response(buffer.slice(0), {
        headers: {
          'Content-Type': 'application/octet-stream',
          'Content-Length': buffer.byteLength.toString(),
        },
      });
      await cache.put(url, cacheResponse);
    } catch (e) {
      console.warn('[Cache] Failed to persist response into Cache API:', e);
    }
  }

  return buffer;
}
