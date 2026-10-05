// Corre la página de un día en Node con un DOM de mentira y el uPlot real, y
// escribe en stdout un resumen JSON de lo que la vista hizo. No dibuja nada: sirve
// para atrapar errores de la lógica (decodificación, niveles, θ, zoom, hooks) sin
// un navegador. Uso: node harness.cjs <index.html> [--drop <tile>] [--pick-last]
//   --drop <tile>: quita ese arreglo de window.VIZ_DATA antes de arrancar la vista.
//   --pick-last: elige la última opción del selector de θ (un θ sin datos).
"use strict";
const fs = require("fs");
const vm = require("vm");

const args = process.argv.slice(2);
const html = fs.readFileSync(args[0], "utf8");
const flag = (name) => args.indexOf(name);
const scripts = [...html.matchAll(/<script(?: id="[^"]*")?>\n([\s\S]*?)\n<\/script>/g)].map((m) => m[1]);
if (scripts.length !== 3) throw new Error("se esperaban 3 <script>, hay " + scripts.length);
const [uplotJs, dataJs, appJs] = scripts;

const W = 1500;
const H = 800;
const drawLog = [];

class HTMLElementStub {}

function element(tag) {
  const el = Object.assign(Object.create(HTMLElementStub.prototype), {
    tagName: String(tag).toUpperCase(),
    children: [],
    listeners: {},
    style: new Proxy({}, { set: (t, k, v) => ((t[k] = v), true), get: (t, k) => (k in t ? t[k] : "") }),
    classList: { add() {}, remove() {}, contains: () => false, toggle() {} },
    textContent: "",
    className: "",
    hidden: false,
    value: "",
    options: [],
    selectedIndex: -1,
    clientWidth: W,
    clientHeight: H,
    offsetWidth: 120,
    offsetHeight: 60,
    appendChild(child) {
      el.children.push(child);
      child.parent = el;
      if (el.tagName === "SELECT") {
        el.options.push(child);
      }
      return child;
    },
    insertBefore(child) { return el.appendChild(child); },
    removeChild() {},
    remove() { el.removed = true; },
    setAttribute() {},
    getAttribute: () => null,
    addEventListener(name, fn) { (el.listeners[name] = el.listeners[name] || []).push(fn); },
    removeEventListener() {},
    dispatch(name, ev) { (el.listeners[name] || []).forEach((fn) => fn(ev || {})); },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: W, height: H, right: W, bottom: H }),
    getContext() {
      return new Proxy(
        {},
        {
          get(t, k) {
            if (k === "measureText") return () => ({ width: 10 });
            if (k in t) return t[k];
            return (...args) => { drawLog.push([String(k), ...args]); };
          },
          set(t, k, v) { t[k] = v; if (k === "fillStyle") drawLog.push(["fillStyle", v]); return true; },
        }
      );
    },
    querySelector: () => null,
    querySelectorAll: () => [],
    focus() {},
    blur() {},
  });
  Object.defineProperty(el, "value", {
    get() { return el._value !== undefined ? el._value : el.options[el.selectedIndex]?.value ?? ""; },
    set(v) { el._value = v; el.selectedIndex = el.options.findIndex((o) => o.value === v); },
  });
  return el;
}

const byId = {};
for (const id of ["s-day", "theta", "s-updated", "s-mode-box", "s-mode", "s-reasons", "panels", "price", "volume", "f-level", "tip", "viz-data"]) {
  byId[id] = element(id === "theta" ? "select" : "div");
}
byId.tip.hidden = true;

const docListeners = {};
const document = {
  getElementById: (id) => byId[id] || null,
  createElement: element,
  createElementNS: element,
  createTextNode: (t) => ({ textContent: t }),
  documentElement: element("html"),
  body: element("body"),
  head: element("head"),
  addEventListener(name, fn) { (docListeners[name] = docListeners[name] || []).push(fn); },
  removeEventListener() {},
  title: "",
  hidden: false,
};

const winListeners = {};
const logs = [];
const sandbox = {
  document,
  navigator: { userAgent: "node-harness", language: "es" },
  HTMLElement: HTMLElementStub,
  CustomEvent: class { constructor(type) { this.type = type; } },
  dispatchEvent() {},
  devicePixelRatio: 1,
  innerWidth: W,
  innerHeight: H,
  performance,
  Path2D: class { moveTo() {} lineTo() {} rect() {} closePath() {} addPath() {} arc() {} },
  requestAnimationFrame: (fn) => setTimeout(fn, 0),
  cancelAnimationFrame: clearTimeout,
  matchMedia: () => ({ addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {}, matches: false }),
  addEventListener(name, fn) { (winListeners[name] = winListeners[name] || []).push(fn); },
  removeEventListener() {},
  getComputedStyle: () => ({ getPropertyValue: () => "" }),
  atob: (s) => Buffer.from(s, "base64").toString("binary"),
  queueMicrotask,
  setTimeout,
  clearTimeout,
  Float64Array, Float32Array, Uint32Array, Int32Array, Uint8Array, Array, Math, Number, Object, JSON, String, Infinity, isFinite, parseInt,
  console: { info: (...a) => logs.push(a.join(" ")), log() {}, warn: (...a) => logs.push("WARN " + a.join(" ")), error: (...a) => logs.push("ERROR " + a.join(" ")) },
};
sandbox.window = sandbox;
vm.createContext(sandbox);

const instances = [];
const errors = [];

// Mensaje y las primeras líneas de la pila, sin volcar el código minificado.
function brief(e) {
  const lines = String(e && e.stack ? e.stack : e).split("\n").filter((l) => /^\s*at |Error/.test(l));
  return lines.slice(0, 8).map((l) => l.slice(0, 160)).join(" | ");
}

function run(code, name) {
  try {
    vm.runInContext(code, sandbox, { filename: name });
  } catch (e) {
    errors.push(name + ": " + brief(e));
  }
}

run(uplotJs, "uPlot");
if (errors.length) { console.error(errors.join("\n")); process.exit(1); }
// Se envuelve el constructor para poder inspeccionar los paneles desde aquí.
vm.runInContext(
  `(function(){ var Real = uPlot; window.__instances = [];
    var Wrapped = function(opts, data, el){ var u = new Real(opts, data, el); window.__instances.push(u); return u; };
    Wrapped.paths = Real.paths; window.uPlot = Wrapped; uPlot = Wrapped; })();`,
  sandbox
);
run(dataJs, "viz-data");
if (flag("--drop") >= 0) vm.runInContext("delete window.VIZ_DATA.files[" + JSON.stringify(args[flag("--drop") + 1]) + "]", sandbox);
const started = Date.now();
run(appJs, "app");

const tick = () => new Promise((r) => setTimeout(r, 5));

function plots() {
  const [price, vol] = sandbox.__instances;
  return { price, vol };
}

function snapshot(label) {
  const { price, vol } = plots();
  const info = { label };
  if (price) {
    info.priceLen = price.data[0].length;
    info.volLen = vol.data[0].length;
    info.x = [price.scales.x.min, price.scales.x.max];
    info.vx = [vol.scales.x.min, vol.scales.x.max];
    info.y = [price.scales.p.min, price.scales.p.max];
    const ys = price.data[1].filter((v) => v !== null);
    info.data = {
      points: price.data[0].length,
      nulls: price.data[1].length - ys.length,
      yMin: Math.min(...ys),
      yMax: Math.max(...ys),
      xLast: price.data[0][price.data[0].length - 1],
      volSum: Array.from(vol.data[1]).reduce((a, b) => a + b, 0),
    };
  }
  info.mode = byId["s-mode"].textContent;
  info.reasons = byId["s-reasons"].textContent;
  info.level = byId["f-level"].textContent;
  info.metrics = JSON.parse(JSON.stringify(sandbox.VIZ_METRICS || {}));
  return info;
}

(async () => {
  const out = { errors, steps: [] };
  await tick();
  await tick();
  out.dataScriptRemoved = !!byId["viz-data"].removed;
  out.day = byId["s-day"].textContent;
  out.updated = byId["s-updated"].textContent;
  out.options = byId.theta.options.map((o) => o.textContent);
  out.steps.push(snapshot("inicio"));
  out.panelHeights = plots().price ? [plots().price.height, plots().vol.height] : null;

  if (plots().price) {
    const { price } = plots();
    if (flag("--pick-last") >= 0) {
      byId.theta.value = byId.theta.options[byId.theta.options.length - 1].value;
      byId.theta.dispatch("change");
      await tick();
      out.steps.push(snapshot("θ-sin-datos"));
      out.messages = drawLog.filter((c) => c[0] === "fillText").map((c) => c[1]);
    }
    // Cambio de θ: solo repinta; el eje Y no se mueve.
    if (byId.theta.options.length > 1) {
      const before = snapshot("antes-θ");
      drawLog.length = 0;
      byId.theta.value = byId.theta.options[1].value;
      byId.theta.dispatch("change");
      await tick();
      out.steps.push(before, snapshot("después-θ"));
      out.regionFills = drawLog.filter((c) => c[0] === "fillRect").length;
    }
    // Cursor sobre una cubeta: el tooltip se arma desde los tiles.
    byId.price.listeners = byId.price.listeners || {};
    const over = price.over;
    over.dispatch("mouseenter");
    price.setCursor({ left: 700, top: 200 });
    await tick();
    out.tooltip = byId.tip.hidden ? null : byId.tip.textContent;
    over.dispatch("mouseleave");
    out.tooltipHidden = byId.tip.hidden;
    // Zoom explícito a una hora: sube el nivel y el rango visible se conserva.
    price.setScale("x", { min: 36000, max: 39600 });
    await tick();
    await tick();
    out.steps.push(snapshot("zoom"));
    // Doble clic: día completo.
    price.over.dispatch("dblclick");
    await tick();
    await tick();
    out.steps.push(snapshot("reset"));
    // Cambio de tamaño de la ventana.
    (winListeners.resize || []).forEach((fn) => fn());
    await tick();
    out.steps.push(snapshot("resize"));
  }
  out.logs = logs;
  out.errors = errors;
  process.stdout.write(JSON.stringify(out));
})().catch((e) => {
  process.stdout.write(JSON.stringify({ errors: [String(e && e.stack ? e.stack : e)], logs }));
});
