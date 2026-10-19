import { chromium } from 'playwright';

async function runTest() {
  console.log('[Test] Launching Chrome with WebGPU support...');
  const browser = await chromium.launch({
    headless: true,
    channel: 'chrome',
    args: [
      '--enable-unsafe-webgpu',
      '--use-angle=metal',
      '--enable-features=Vulkan,UseSkiaRenderer',
      '--no-sandbox',
    ],
  });

  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 },
  });

  const page = await context.newPage();

  // Capture all console logs
  page.on('console', (msg) => {
    console.log(`[Browser Console ${msg.type()}]:`, msg.text());
  });

  page.on('pageerror', (err) => {
    console.error('[Browser PageError]:', err);
  });

  console.log('[Test] Navigating to http://127.0.0.1:5173/ ...');
  await page.goto('http://127.0.0.1:5173/', { waitUntil: 'networkidle' });

  console.log('[Test] Page loaded. Title:', await page.title());

  // Click START GAME button
  console.log('[Test] Clicking "START GAME" button...');
  const startBtn = page.locator('button:has-text("START GAME")');
  await startBtn.click();

  // Wait for loading progress or frame rendering
  console.log('[Test] Waiting for engine warmup and first frames...');
  
  let frameRendered = false;
  for (let i = 0; i < 60; i++) {
    await page.waitForTimeout(1000);
    const canvasData = await page.evaluate(() => {
      const canvas = document.querySelector('canvas');
      if (!canvas) return null;
      const ctx = canvas.getContext('2d');
      if (!ctx || canvas.width === 0 || canvas.height === 0) return null;
      const data = ctx.getImageData(0, 0, Math.min(canvas.width, 10), Math.min(canvas.height, 10)).data;
      let nonZero = 0;
      for (let j = 0; j < data.length; j++) {
        if (data[j] > 0) nonZero++;
      }
      return { width: canvas.width, height: canvas.height, nonZero, length: data.length };
    });

    if (canvasData && canvasData.nonZero > 0) {
      console.log(`[Test] Frame rendered! Canvas size: ${canvasData.width}x${canvasData.height}, non-zero pixels: ${canvasData.nonZero}/${canvasData.length}`);
      frameRendered = true;
      break;
    }
  }

  // Take a screenshot
  await page.screenshot({ path: 'client_game_screenshot.png' });
  console.log('[Test] Saved screenshot to client_game_screenshot.png');

  if (!frameRendered) {
    throw new Error('Frame was not rendered within timeout');
  }

  // Simulate key actions
  console.log('[Test] Simulating player inputs: "d" (Right), "w" (Jump), "k" (Attack)...');
  await page.keyboard.down('d');
  await page.waitForTimeout(500);
  await page.keyboard.up('d');

  await page.keyboard.down('w');
  await page.waitForTimeout(300);
  await page.keyboard.up('w');

  await page.keyboard.down('k');
  await page.waitForTimeout(200);
  await page.keyboard.up('k');

  await page.screenshot({ path: 'client_game_after_input.png' });
  console.log('[Test] Saved screenshot after input to client_game_after_input.png');

  // Test toggling Upscaler to ON (288p)
  console.log('[Test] Toggling Upscaler to ON (288p)...');
  const upscalerOnBtn = page.locator('button:has-text("ON (288p)")');
  await upscalerOnBtn.click();
  await page.waitForTimeout(2000);

  const hrCanvasData = await page.evaluate(() => {
    const canvas = document.querySelector('canvas');
    return canvas ? { width: canvas.width, height: canvas.height } : null;
  });
  console.log(`[Test] Canvas after Upscaler ON: ${hrCanvasData?.width}x${hrCanvasData?.height}`);
  if (hrCanvasData?.width !== 512 || hrCanvasData?.height !== 288) {
    throw new Error(`Expected 512x288 after upscaler toggle, got ${hrCanvasData?.width}x${hrCanvasData?.height}`);
  }

  // Test toggling Sampling Steps to 3 Steps (Quality)
  console.log('[Test] Toggling Sampling Steps to 3 Steps (Quality)...');
  const steps3Btn = page.locator('button:has-text("3 Steps (Quality)")');
  await steps3Btn.click();
  await page.waitForTimeout(2000);

  // Take final screenshot with upscaler ON and 3 steps
  await page.screenshot({ path: 'client_game_upscaled.png' });
  console.log('[Test] Saved screenshot to client_game_upscaled.png');

  await browser.close();
  console.log('[Test] Browser test passed successfully!');
}

runTest().catch((err) => {
  console.error('[Test Failed]:', err);
  process.exit(1);
});
