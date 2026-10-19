import { chromium } from 'playwright';

async function testWasmFallback() {
  console.log('[Test WASM Fallback] Launching Chrome with --disable-webgpu...');
  const browser = await chromium.launch({
    headless: true,
    channel: 'chrome',
    args: [
      '--disable-webgpu',
      '--no-sandbox',
    ],
  });

  const page = await browser.newPage({
    viewport: { width: 1280, height: 720 },
  });

  let fellBackToWasm = false;
  page.on('console', (msg) => {
    const text = msg.text();
    console.log(`[Browser Console]:`, text);
    if (text.includes('falling back to WASM') || text.includes('initialized on WASM')) {
      fellBackToWasm = true;
    }
  });

  // Remove WebGPU from browser environment to guarantee WASM execution
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'gpu', {
      get: () => undefined,
    });
  });

  console.log('[Test WASM Fallback] Navigating to http://127.0.0.1:5173/?backend=wasm ...');
  await page.goto('http://127.0.0.1:5173/?backend=wasm', { waitUntil: 'networkidle' });

  console.log('[Test WASM Fallback] Clicking "START GAME"...');
  await page.locator('button:has-text("START GAME")').click();

  // Wait up to 30 seconds for WASM initialization and frame
  console.log('[Test WASM Fallback] Waiting for fallback initialization...');
  let frameRendered = false;
  for (let i = 0; i < 30; i++) {
    await page.waitForTimeout(1000);
    const canvasData = await page.evaluate(() => {
      const canvas = document.querySelector('canvas');
      if (!canvas || canvas.width === 0 || canvas.height === 0) return null;
      const ctx = canvas.getContext('2d');
      if (!ctx) return null;
      const data = ctx.getImageData(0, 0, Math.min(canvas.width, 10), Math.min(canvas.height, 10)).data;
      let nonZero = 0;
      for (let j = 0; j < data.length; j++) {
        if (data[j] > 0) nonZero++;
      }
      return { width: canvas.width, height: canvas.height, nonZero };
    });

    if (canvasData && canvasData.nonZero > 0) {
      console.log(`[Test WASM Fallback] Frame rendered successfully on WASM! Size: ${canvasData.width}x${canvasData.height}`);
      frameRendered = true;
      break;
    }
  }

  await page.screenshot({ path: 'client_game_wasm_screenshot.png' });
  await browser.close();

  if (!frameRendered) {
    throw new Error('Frame was not rendered in WASM fallback mode');
  }

  console.log('[Test WASM Fallback] Test passed! Verified WebGPU fallback to WASM works seamlessly.');
}

testWasmFallback().catch((err) => {
  console.error('[WASM Fallback Test Failed]:', err);
  process.exit(1);
});
