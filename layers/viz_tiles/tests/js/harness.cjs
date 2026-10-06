// Corre la página de un día en Node con un DOM de mentira y el uPlot real, y
// escribe en stdout un resumen JSON de lo que la vista hizo. No dibuja nada: sirve
// para atrapar errores de la lógica (decodificación, marcos por píxel, θ, zoom, hooks)
// sin un navegador, y deja las marcas (fillRect) que cada panel dibujó. Uso:
//   node harness.cjs <index.html> [--drop <archivo>] [--corrupt <archivo>] [--pick-last]
//   --drop <archivo>: quita ese archivo de window.VIZ_DATA antes de arrancar la vista.
//   --corrupt <archivo>: lo deja con 6 bytes (8 caracteres de base64): presente, de tamaño inesperado.
//   --pick-last: elige la última opción del selector de θ (un θ sin datos).
//   --theta <k>: antes de los zooms y los cursores, el θ de la opción k (por defecto queda el de la opción 1).
//   --zoom <min>,<max>: un zoom explícito a ese rango (en segundos); se puede repetir.
//   --sweep: pasa el cursor por todo el ancho con cada θ y devuelve los textos de θ del tooltip.
//   --nav <k>: con el θ de la opción k, "evento siguiente" dos veces, "anterior" y "ajustar a la ventana".
//   --nav-switch <j>: tras navegar, cambia al θ de la opción j y deja el paso "nav-switch".
//   --nav-at <seg>: antes de navegar, un zoom de 600 s centrado en ese segundo (la escala "fijada" por el humano).
//   --hover <panel>@<seg>[@<theta>]: tras el último zoom, el cursor de ese panel (price, confirms o volume)
//     en ese segundo, con el θ de la opción pedida; devuelve el tooltip. Se puede repetir.
//   --bench <n>: n zooms alternados (día completo y un tramo del medio) y los tiempos de redibujo.
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
const textLog = []; // todo fillText de la corrida; drawLog se vacía en algunos pasos

class HTMLElementStub {}

const canvases = []; // un lienzo por panel de uPlot, en el orden en que se crearon

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
      // Cada lienzo guarda lo último que se dibujó en él: `clearRect` empieza un dibujo nuevo.
      el.log = [];
      return new Proxy(
        {},
        {
          get(t, k) {
            if (k === "measureText") return () => ({ width: 10 });
            if (k in t) return t[k];
            return (...args) => {
              if (k === "clearRect") el.log.length = 0;
              drawLog.push([String(k), ...args]);
              el.log.push([String(k), ...args]);
              if (k === "fillText") textLog.push(args[0]);
            };
          },
          set(t, k, v) {
            t[k] = v;
            if (["fillStyle", "strokeStyle", "lineWidth"].includes(k)) {
              drawLog.push([k, v]);
              el.log.push([k, v]);
            }
            return true;
          },
        }
      );
    },
    querySelector: () => null,
    querySelectorAll: () => [],
    focus() {},
    blur() {},
  });
  if (el.tagName === "CANVAS") canvases.push(el);
  Object.defineProperty(el, "value", {
    get() { return el._value !== undefined ? el._value : el.options[el.selectedIndex]?.value ?? ""; },
    set(v) { el._value = v; el.selectedIndex = el.options.findIndex((o) => o.value === v); },
  });
  return el;
}

const byId = {};
for (const id of ["s-day", "theta", "s-updated", "s-data-box", "s-data", "panels", "price", "confirms", "volume", "f-view", "tip", "viz-data", "ev-prev", "ev-next", "ev-fit", "ev-info"]) {
  byId[id] = element(id === "theta" ? "select" : id.startsWith("ev-") && id !== "ev-info" ? "button" : "div");
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
  // Guarda los trazos de moveTo/lineTo para poder leer la geometría de un camino.
  Path2D: class {
    constructor() { this.ops = []; }
    moveTo(x, y) { this.ops.push(["moveTo", x, y]); }
    lineTo(x, y) { this.ops.push(["lineTo", x, y]); }
    rect(x, y, w, h) { this.ops.push(["rect", x, y, w, h]); }
    closePath() {} addPath() {} arc() {}
  },
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
if (flag("--corrupt") >= 0) {
  const name = JSON.stringify(args[flag("--corrupt") + 1]);
  vm.runInContext("window.VIZ_DATA.files[" + name + "] = window.VIZ_DATA.files[" + name + "].slice(0, 8)", sandbox);
}
const started = Date.now();
run(appJs, "app");

const tick = () => new Promise((r) => setTimeout(r, 5));

function plots() {
  const [price, conf, vol] = sandbox.__instances;
  return { price, conf, vol };
}

// Los rectángulos que un lienzo dibujó con ese estilo de relleno, tal como los dejó el último dibujo.
function canvasOf(plot) {
  return canvases[sandbox.__instances.indexOf(plot)];
}

function marksOf(plot, style) {
  const out = [];
  let current = null;
  for (const call of canvasOf(plot).log || []) {
    if (call[0] === "fillStyle") current = call[1];
    else if (call[0] === "fillRect" && current === style) out.push(call.slice(1));
  }
  return out;
}

const COLORS = {
  price: "#e6edf3",
  volume: "rgba(110, 138, 168, 0.85)",
  confirms: "rgba(86, 182, 194, 0.42)",
  simul: "rgb(86, 182, 194)",
};

function snapshot(label) {
  const { price, conf, vol } = plots();
  const info = { label };
  if (price) {
    info.x = [price.scales.x.min, price.scales.x.max];
    info.vx = [vol.scales.x.min, vol.scales.x.max];
    info.cx = [conf.scales.x.min, conf.scales.x.max];
    info.yPrice = [price.scales.p.min, price.scales.p.max];
    info.yConf = [conf.scales.c.min, conf.scales.c.max];
    info.yVol = [vol.scales.v.min, vol.scales.v.max];
    info.plotW = price.bbox.width / sandbox.devicePixelRatio;
    info.plot = { left: price.bbox.left, top: price.bbox.top, width: price.bbox.width, height: price.bbox.height };
    info.plotConf = { left: conf.bbox.left, top: conf.bbox.top, width: conf.bbox.width, height: conf.bbox.height };
    info.plotVol = { left: vol.bbox.left, top: vol.bbox.top, width: vol.bbox.width, height: vol.bbox.height };
    info.marks = {
      price: marksOf(price, COLORS.price),
      volume: marksOf(vol, COLORS.volume),
      confirms: marksOf(conf, COLORS.confirms),
      simul: marksOf(conf, COLORS.simul),
    };
    info.panelTexts = {
      price: canvasOf(price).log.filter((c) => c[0] === "fillText").map((c) => c[1]),
      confirms: canvasOf(conf).log.filter((c) => c[0] === "fillText").map((c) => c[1]),
      volume: canvasOf(vol).log.filter((c) => c[0] === "fillText").map((c) => c[1]),
    };
  }
  info.status = byId["s-data"].textContent;
  info.statusClass = byId["s-data-box"].className;
  info.view = byId["f-view"].textContent;
  info.metrics = JSON.parse(JSON.stringify(sandbox.VIZ_METRICS || {}));
  info.nav = { info: byId["ev-info"].textContent, prev: !!byId["ev-prev"].disabled, next: !!byId["ev-next"].disabled, fit: !!byId["ev-fit"].disabled };
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
  // Los textos del primer trazo de los dos paneles, antes de que algo limpie el registro.
  out.firstDraw = drawLog.filter((c) => ["fillStyle", "fillRect", "strokeStyle", "lineWidth", "setLineDash", "moveTo", "lineTo"].includes(c[0]));
  out.firstMessages = drawLog.filter((c) => c[0] === "fillText").map((c) => c[1]);
  out.panelHeights = plots().price ? [plots().price.height, plots().conf.height, plots().vol.height] : null;
  out.regionDraw = drawLog.filter((c) => ["fillStyle", "fillRect", "strokeRect", "lineWidth", "strokeStyle"].includes(c[0]));

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
    // Navegación por eventos con el θ pedido.
    if (flag("--nav") >= 0) {
      const k = Number(args[flag("--nav") + 1]);
      byId.theta.value = byId.theta.options[k].value;
      byId.theta.dispatch("change");
      await tick();
      if (flag("--nav-at") >= 0) {
        const at = Number(args[flag("--nav-at") + 1]);
        price.setScale("x", { min: at - 300, max: at + 300 });
        await tick();
        await tick();
      }
      out.steps.push(snapshot("nav-0"));
      for (const [label, id] of [["nav-1", "ev-next"], ["nav-2", "ev-next"], ["nav-3", "ev-prev"], ["nav-fit", "ev-fit"]]) {
        drawLog.length = 0;
        byId[id].dispatch("click");
        await tick();
        await tick();
        out.steps.push(snapshot(label));
        if (label === "nav-fit") out.fitDraw = drawLog.filter((c) => ["fillStyle", "fillRect", "strokeRect"].includes(c[0]));
      }
      if (flag("--nav-switch") >= 0) {
        byId.theta.value = byId.theta.options[Number(args[flag("--nav-switch") + 1])].value;
        byId.theta.dispatch("change");
        await tick();
        out.steps.push(snapshot("nav-switch"));
      }
      out.navTexts = [...new Set(textLog)];
    }
    // Cursor sobre un píxel: el tooltip se arma desde los ticks.
    const over = price.over;
    over.dispatch("mouseenter");
    price.setCursor({ left: 700, top: 200 });
    await tick();
    out.tooltip = byId.tip.hidden ? null : byId.tip.textContent;
    over.dispatch("mouseleave");
    out.tooltipHidden = byId.tip.hidden;
    // Zoom explícito a una hora y de vuelta al día entero (doble clic).
    price.setScale("x", { min: 36000, max: 39600 });
    await tick();
    await tick();
    out.steps.push(snapshot("zoom"));
    price.over.dispatch("dblclick");
    await tick();
    await tick();
    out.steps.push(snapshot("reset"));
    // Cambio de tamaño de la ventana.
    (winListeners.resize || []).forEach((fn) => fn());
    await tick();
    out.steps.push(snapshot("resize"));
    // El θ pedido para los zooms y los cursores que siguen.
    if (flag("--theta") >= 0) {
      byId.theta.value = byId.theta.options[Number(args[flag("--theta") + 1])].value;
      byId.theta.dispatch("change");
      await tick();
    }
    // Zooms explícitos pedidos por la prueba.
    for (let i = 0; i < args.length; i++) {
      if (args[i] !== "--zoom") continue;
      const [min, max] = args[i + 1].split(",").map(Number);
      price.setScale("x", { min, max });
      await tick();
      await tick();
      out.steps.push(snapshot("zoom:" + args[i + 1]));
    }
    // El cursor de un panel en un segundo del día, con el θ pedido: el tooltip que arma la vista.
    out.hovers = [];
    for (let i = 0; i < args.length; i++) {
      if (args[i] !== "--hover") continue;
      const [panel, sec, theta] = args[i + 1].split("@");
      if (theta !== undefined) {
        byId.theta.value = byId.theta.options[Number(theta)].value;
        byId.theta.dispatch("change");
        await tick();
      }
      const plot = plots()[{ price: "price", confirms: "conf", volume: "vol" }[panel]];
      plot.over.dispatch("mouseenter");
      plot.setCursor({ left: plot.valToPos(Number(sec), "x"), top: 20 });
      await tick();
      out.hovers.push({ spec: args[i + 1], tooltip: byId.tip.hidden ? null : byId.tip.textContent });
      plot.over.dispatch("mouseleave");
    }
    // Los textos de θ del tooltip en todo el ancho y con cada θ: ningún estado lleva glifo.
    if (flag("--sweep") >= 0) {
      const texts = new Set();
      price.over.dispatch("mouseenter");
      for (let k = 0; k < byId.theta.options.length; k++) {
        byId.theta.value = byId.theta.options[k].value;
        byId.theta.dispatch("change");
        for (let left = 40; left < 1440; left += 20) {
          price.setCursor({ left, top: 200 });
          if (!byId.tip.hidden) byId.tip.textContent.split("\n").filter((l) => l.startsWith("θ ")).forEach((l) => texts.add(l));
        }
      }
      price.over.dispatch("mouseleave");
      out.thetaLines = [...texts];
    }
    // Tiempos de redibujo: zooms alternados entre el día completo y un tramo del medio.
    if (flag("--bench") >= 0) {
      const n = Number(args[flag("--bench") + 1]);
      out.bench = [];
      for (let i = 0; i < n; i++) {
        const range = i % 2 ? [0, 86400] : [30000 + i * 7, 40000 + i * 7];
        price.setScale("x", { min: range[0], max: range[1] });
        await tick();
        await tick();
        const m = JSON.parse(JSON.stringify(sandbox.VIZ_METRICS));
        out.bench.push({ range, draw_ms: m.draw_ms, frame_ms: m.frame.ms, ticks: m.frame.ticks });
      }
    }
  }
  out.texts = [...new Set(textLog)];
  out.logs = logs;
  out.errors = errors;
  process.stdout.write(JSON.stringify(out));
})().catch((e) => {
  process.stdout.write(JSON.stringify({ errors: [String(e && e.stack ? e.stack : e)], logs }));
});
