import { spawnSync } from "node:child_process";
import {
  chmodSync,
  closeSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  openSync,
  readSync,
  readdirSync,
  statSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const desktopDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = resolve(desktopDir, "..");

function fail(message) {
  console.error(`prepare-sidecar: ${message}`);
  process.exit(1);
}

const rustc = spawnSync("rustc", ["-vV"], { encoding: "utf8" });
if (rustc.status !== 0) {
  fail("Rust/Cargo is required to determine the Tauri target triple.");
}
const targetTriple = rustc.stdout.match(/^host:\s+(.+)$/m)?.[1]?.trim();
if (!targetTriple) {
  fail("Unable to determine the Rust host target triple.");
}

const workerFilename = process.platform === "win32" ? "slidebridge-worker.exe" : "slidebridge-worker";
const workerSource = join(repoRoot, "dist", workerFilename);
const workerTarget = join(
  desktopDir,
  "src-tauri",
  "binaries",
  `slidebridge-worker-${targetTriple}${process.platform === "win32" ? ".exe" : ""}`,
);

function isNativeExecutable(path) {
  if (!existsSync(path) || statSync(path).size < 4) {
    return false;
  }
  const header = Buffer.alloc(4);
  const descriptor = openSync(path, "r");
  try {
    readSync(descriptor, header, 0, header.length, 0);
  } finally {
    closeSync(descriptor);
  }
  if (process.platform === "win32") {
    return header[0] === 0x4d && header[1] === 0x5a;
  }
  if (process.platform === "darwin") {
    return new Set(["cffaedfe", "feedfacf", "cafebabe", "bebafeca"]).has(header.toString("hex"));
  }
  return header.toString("hex") === "7f454c46";
}

function collectPythonSources(directory) {
  const files = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...collectPythonSources(path));
    } else if (entry.isFile() && entry.name.endsWith(".py")) {
      files.push(path);
    }
  }
  return files;
}

const buildInputs = [
  join(repoRoot, "SlideBridgeWorker.spec"),
  join(repoRoot, "THIRD_PARTY_NOTICES.md"),
  join(repoRoot, "requirements.txt"),
  join(repoRoot, "worker_main.py"),
  ...collectPythonSources(join(repoRoot, "ibl2svs")),
];
if (process.env.IBL2SVS_TURBOJPEG && existsSync(process.env.IBL2SVS_TURBOJPEG)) {
  buildInputs.push(process.env.IBL2SVS_TURBOJPEG);
}
const newestInput = Math.max(...buildInputs.map((path) => statSync(path).mtimeMs));
if (isNativeExecutable(workerTarget) && statSync(workerTarget).mtimeMs >= newestInput) {
  console.log(`prepare-sidecar: using current ${workerTarget}`);
  process.exit(0);
}

const candidates = [];
if (process.env.PYTHON) {
  candidates.push({ command: process.env.PYTHON, prefix: [] });
}
if (process.platform === "win32") {
  candidates.push({ command: "python", prefix: [] }, { command: "py", prefix: ["-3"] });
} else {
  candidates.push({ command: "python3", prefix: [] }, { command: "python", prefix: [] });
}

const seen = new Set();
const python = candidates.find(({ command, prefix }) => {
  const key = `${command}\0${prefix.join("\0")}`;
  if (seen.has(key)) {
    return false;
  }
  seen.add(key);
  return spawnSync(command, [...prefix, "-c", "import PyInstaller"], { stdio: "ignore" }).status === 0;
});
if (!python) {
  fail("No Python installation with PyInstaller was found. Install requirements.txt or set PYTHON to the correct interpreter.");
}

console.log(`prepare-sidecar: building ${workerFilename} with ${python.command}`);
const build = spawnSync(
  python.command,
  [...python.prefix, "-m", "PyInstaller", "SlideBridgeWorker.spec", "--clean", "--noconfirm"],
  { cwd: repoRoot, stdio: "inherit" },
);
if (build.status !== 0) {
  process.exit(build.status ?? 1);
}
if (!isNativeExecutable(workerSource)) {
  fail(`PyInstaller did not create a native executable at ${workerSource}.`);
}

mkdirSync(dirname(workerTarget), { recursive: true });
copyFileSync(workerSource, workerTarget);
if (process.platform !== "win32") {
  chmodSync(workerTarget, 0o755);
}
console.log(`prepare-sidecar: wrote ${workerTarget}`);
