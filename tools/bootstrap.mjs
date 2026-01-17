import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

function run(cmd, args, opts = {}) {
  const result = spawnSync(cmd, args, {
    stdio: 'inherit',
    shell: false,
    ...opts,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`Command failed (${result.status}): ${cmd} ${args.join(' ')}`);
  }
}

function tryRunCapture(cmd, args) {
  const result = spawnSync(cmd, args, { stdio: ['ignore', 'pipe', 'ignore'], shell: false });
  if (result.status === 0) return String(result.stdout || '').trim();
  return null;
}

function findPython() {
  const isWin = process.platform === 'win32';

  // Prefer py launcher on Windows (lets us pin 3.12 if installed).
  if (isWin) {
    const exe = tryRunCapture('py', ['-3.12', '-c', 'import sys; print(sys.executable)']);
    if (exe) return { cmd: 'py', baseArgs: ['-3.12'] };

    const anyPy = tryRunCapture('py', ['-c', 'import sys; print(sys.executable)']);
    if (anyPy) return { cmd: 'py', baseArgs: [] };
  }

  // Fall back to python3/python.
  const py3 = tryRunCapture('python3', ['-c', 'import sys; print(sys.executable)']);
  if (py3) return { cmd: 'python3', baseArgs: [] };

  const py = tryRunCapture('python', ['-c', 'import sys; print(sys.executable)']);
  if (py) return { cmd: 'python', baseArgs: [] };

  throw new Error(
    'Python not found. Install Python 3.12+ and ensure `python` (or `python3`, or Windows `py`) is on PATH.'
  );
}

function venvPythonPath(venvDir) {
  if (process.platform === 'win32') {
    return path.join(venvDir, 'Scripts', 'python.exe');
  }
  return path.join(venvDir, 'bin', 'python');
}

function main() {
  const repoRoot = process.cwd();
  const venvDir = path.join(repoRoot, '.venv');

  const { cmd, baseArgs } = findPython();

  if (!fs.existsSync(venvDir)) {
    console.log('Creating virtual environment (.venv)...');
    run(cmd, [...baseArgs, '-m', 'venv', '.venv'], { cwd: repoRoot });
  } else {
    console.log('Virtual environment already exists (.venv).');
  }

  const vpy = venvPythonPath(venvDir);
  if (!fs.existsSync(vpy)) {
    throw new Error(`Expected venv python not found: ${vpy}`);
  }

  console.log('Upgrading pip...');
  run(vpy, ['-m', 'pip', 'install', '--upgrade', 'pip'], { cwd: repoRoot });

  console.log('Installing Python dependencies (requirements.txt)...');
  run(vpy, ['-m', 'pip', 'install', '-r', 'requirements.txt'], { cwd: repoRoot });

  const cfg = path.join(repoRoot, 'config.json');
  const cfgExample = path.join(repoRoot, 'config.example.json');
  if (!fs.existsSync(cfg) && fs.existsSync(cfgExample)) {
    console.log('\nConfig:');
    console.log('- Option A (recommended): run controller discovery:');
    console.log('  - npm run setup  (already done)');
    console.log('  - then: node is not required; run:');
    console.log('    - ' + (process.platform === 'win32' ? '.\\.venv\\Scripts\\python.exe' : './.venv/bin/python') + ' tools/discover_controller.py --write');
    console.log('- Option B: copy config.example.json -> config.json and fill it in (config.json stays local-only).');
  }

  console.log('\nDone. Next:');
  console.log('- Start HTTP server: npm run start');
  console.log('- Start STDIO server: npm run start:stdio');
  console.log('- Run end-to-end checks: npm run e2e');

  if (process.platform !== 'win32') {
    console.log('\nNote: If your system only has Python as `python3`, this script already handled that.');
  }
}

main();
