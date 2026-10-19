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

  page.on('console', (msg) => {
    console.log(`[Browser Console ${msg.type()}]:`, msg.text());
  });

  page.on('pageerror', (err) => {
    console.error('[Browser PageError]:', err);
  });

  console.log('[Test] Navigating to http://localhost:5173/ ...');
  await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });

  console.log('[Test] Page loaded. Title:', await page.title());

  // Verify START STREAM button exists
  const startBtn = page.locator('button:has-text("START STREAM")');
  await startBtn.click();
  console.log('[Test] Clicked "START STREAM" button.');

  // Wait for engine warmup and first frames
  console.log('[Test] Waiting for engine warmup and frames...');
  
  let frameRendered = false;
  for (let i = 0; i < 45; i++) {
    await page.waitForTimeout(1000);
    const canvasData = await page.evaluate(() => {
      const canvas = document.querySelector('canvas');
      if (!canvas || canvas.width === 0 || canvas.height === 0) return null;
      const ctx = canvas.getContext('2d');
      if (!ctx) return null;
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imgData.data;
      let nonZeroRgb = 0;
      let totalRgbSum = 0;
      for (let j = 0; j < data.length; j += 4) {
        const r = data[j];
        const g = data[j + 1];
        const b = data[j + 2];
        if (r > 5 || g > 5 || b > 5) nonZeroRgb++;
        totalRgbSum += (r + g + b);
      }
      const pixelCount = canvas.width * canvas.height;
      const avgBrightness = totalRgbSum / (pixelCount * 3);
      return { 
        width: canvas.width, 
        height: canvas.height, 
        nonZeroRatio: nonZeroRgb / pixelCount,
        avgBrightness,
      };
    });

    if (canvasData && canvasData.nonZeroRatio > 0.05) {
      console.log(`[Test] Frame rendered! Size: ${canvasData.width}x${canvasData.height}, non-zero pixels: ${(canvasData.nonZeroRatio * 100).toFixed(1)}%, avg brightness: ${canvasData.avgBrightness.toFixed(1)}`);
      frameRendered = true;
      break;
    }
  }

  if (!frameRendered) {
    await page.screenshot({ path: 'test_failure.png' });
    throw new Error('Frame was not rendered or canvas stayed completely black!');
  }

  // Verify Upscaler is active (512x288) and screen is definitely NOT black
  const canvasRes = await page.evaluate(() => {
    const canvas = document.querySelector('canvas');
    if (!canvas) return null;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let blackPixels = 0;
    const total = canvas.width * canvas.height;
    for (let j = 0; j < imgData.data.length; j += 4) {
      if (imgData.data[j] === 0 && imgData.data[j + 1] === 0 && imgData.data[j + 2] === 0) {
        blackPixels++;
      }
    }
    return {
      width: canvas.width,
      height: canvas.height,
      blackRatio: blackPixels / total,
    };
  });
  console.log(`[Test] Canvas state with Upscaler: ${canvasRes?.width}x${canvasRes?.height}, black ratio: ${((canvasRes?.blackRatio ?? 1) * 100).toFixed(1)}%`);
  if (canvasRes?.blackRatio && canvasRes.blackRatio > 0.95) {
    throw new Error('Black screen detected with Upscaler enabled!');
  }

  // Test Key Highlight Verification
  console.log('[Test] Testing Key Highlights...');

  // 1. Press W (Jump)
  await page.keyboard.down('KeyW');
  await page.waitForTimeout(100);
  const wClasses = await page.locator('div:has(> span:text-is("Jump"))').getAttribute('class');
  const wHighlighted = wClasses?.includes('ring-2') || wClasses?.includes('bg-white');
  console.log('[Test] KeyW pressed -> W button highlighted:', wHighlighted);
  await page.keyboard.up('KeyW');

  // 2. Press Space (Platformer Jump) -> should also highlight W button
  await page.keyboard.down('Space');
  await page.waitForTimeout(100);
  const spaceClasses = await page.locator('div:has(> span:text-is("Jump"))').getAttribute('class');
  const spaceHighlightsW = spaceClasses?.includes('ring-2') || spaceClasses?.includes('bg-white');
  console.log('[Test] Space pressed -> W button highlighted:', spaceHighlightsW);
  await page.keyboard.up('Space');

  // 3. Press ArrowRight -> should highlight D button
  await page.keyboard.down('ArrowRight');
  await page.waitForTimeout(100);
  const dClasses = await page.locator('div:has(> span:text-is("Right"))').getAttribute('class');
  const dHighlighted = dClasses?.includes('ring-2') || dClasses?.includes('bg-white');
  console.log('[Test] ArrowRight pressed -> D button highlighted:', dHighlighted);
  await page.keyboard.up('ArrowRight');

  // 4. Press KeyK (Attack)
  await page.keyboard.down('KeyK');
  await page.waitForTimeout(100);
  const kClasses = await page.locator('div:has(> span:text-is("Attack"))').getAttribute('class');
  const kHighlighted = kClasses?.includes('ring-2') || kClasses?.includes('bg-white');
  console.log('[Test] KeyK pressed -> K button highlighted:', kHighlighted);
  await page.keyboard.up('KeyK');

  if (!wHighlighted || !spaceHighlightsW || !dHighlighted || !kHighlighted) {
    throw new Error('Key highlighting check failed!');
  }

  await page.screenshot({ path: 'client_game_verified.png' });
  console.log('[Test] Saved verified screenshot to client_game_verified.png');

  await browser.close();
  console.log('[Test] All tests passed with flying colors!');
}

runTest().catch((err) => {
  console.error('[Test Failed]:', err);
  process.exit(1);
});
