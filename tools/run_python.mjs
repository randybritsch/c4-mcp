import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

function venvPythonPath(repoRoot) {
  const venvDir = path.join(repoRoot, '.venv');
  if (process.platform === 'win32') {
    return path.join(venvDir, 'Scripts', 'python.exe');
  }
  return path.join(venvDir, 'bin', 'python');
}

function main() {
  const repoRoot = process.cwd();
  const target = process.argv[2];
  const args = process.argv.slice(3);

  if (!target) {
    console.error('Usage: node tools/run_python.mjs <script.py> [args...]');
    process.exit(2);
  }

  const py = venvPythonPath(repoRoot);
  if (!fs.existsSync(py)) {
    console.error('Missing .venv. Run: npm run setup');
    process.exit(2);
  }

  const scriptPath = path.isAbsolute(target) ? target : path.join(repoRoot, target);
  const result = spawnSync(py, [scriptPath, ...args], { stdio: 'inherit', shell: false, cwd: repoRoot });
  process.exit(result.status ?? 1);
}

main();
