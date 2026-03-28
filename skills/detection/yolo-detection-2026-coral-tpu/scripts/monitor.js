/**
 * Coral TPU Monitor (Native)
 * Host-side wrapper to launch the Python detection script directly
 * using the natively built virtual environment.
 */

const { spawn } = require('node:child_process');
const path = require('node:path');
const os = require('node:os');

function main() {
  const skillRoot = path.join(__dirname, '..');
  
  // Determine Python executable inside the venv
  const isWindows = os.platform() === 'win32';
  const pythonCmd = isWindows 
    ? path.join(skillRoot, 'venv', 'Scripts', 'python.exe')
    : path.join(skillRoot, 'venv', 'bin', 'python3');

  const args = [path.join(skillRoot, 'scripts', 'detect.py')];

  // We no longer need volume mapping, the python script accesses
  // the host's raw /tmp/aegis-detection-frames directories directly!

  const env = { ...process.env };
  if (!env.PYTHONUNBUFFERED) {
    env.PYTHONUNBUFFERED = '1';
  }

  const child = spawn(pythonCmd, args, {
    stdio: 'inherit',
    cwd: skillRoot,
    env
  });

  child.on('error', (err) => {
    console.error(`[coral-monitor] Failed to start native python process: ${err.message}`);
    process.exit(1);
  });

  child.on('exit', (code, signal) => {
    console.log(`[coral-monitor] Python process exited with code ${code} (signal ${signal})`);
    process.exit(code || 0);
  });
}

main();
