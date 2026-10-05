/* Vista de un día (TRD-viz §6.6, §6.7). Sin red: los tiles llegan en window.VIZ_DATA,
 * en base64 bajo su nombre. El navegador solo decodifica y hace las dos conversiones
 * de §7.3 (t / 1000 y p / price_scale, o null en el centinela); todo lo demás sale
 * de los tiles tal cual. Sin telemetría: las métricas van a la consola. */
(function () {
  "use strict";

  var DAY_S = 86400;
  var EMPTY = -2147483648; // centinela de p en una columna sin ticks (§7.3)
  var KNOWN_MAJOR = 1;
  var MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace';
  var C = {
    text: "#d7dee6",
    muted: "#9aa7b4",
    grid: "rgba(154, 167, 180, 0.16)",
    price: "#e6edf3",
    volume: "rgba(110, 138, 168, 0.85)",
    amber: "#e3b341",
    up: "63, 185, 80",
    down: "248, 81, 73",
  };
  // Estados de dirección (§7.4): texto, sentido (+1 alza, -1 baja) y fase.
  var STATES = {
    0: { text: "sin evento", sign: 0, strong: false },
    1: { text: "confirmación alza ▲", sign: 1, strong: false },
    2: { text: "overshoot alza ▲", sign: 1, strong: true },
    3: { text: "confirmación baja ▼", sign: -1, strong: false },
    4: { text: "overshoot baja ▼", sign: -1, strong: true },
  };

  var started = performance.now();
  var metrics = (window.VIZ_METRICS = {
    decoded_bytes: 0,
    decode_ms: 0,
    first_paint_ms: null,
    theta_change_ms: [],
  });

  function $(id) {
    return document.getElementById(id);
  }

  var data = window.VIZ_DATA;
  var dataScript = $("viz-data");
  if (dataScript) dataScript.remove(); // el texto base64 deja de vivir también en el DOM

  if (!data || !data.index) {
    $("panels").textContent = "Sin datos: la página no trae tiles.";
    return;
  }

  var index = data.index;
  var files = data.files;
  var scale = index.price_scale;
  var decimals = Math.round(Math.log10(scale));
  var levels = index.levels.slice().sort(function (a, b) { return a - b; });
  var thetas = index.thetas;
  var missingThetas = index.missing_thetas;

  /* ---------- Modo degradado (principio 6) ---------- */

  var reasons = {}; // clave -> texto; si hay alguna, el modo es degradado

  function setReason(key, text) {
    if (text) reasons[key] = text;
    else delete reasons[key];
  }

  function renderStatus() {
    var keys = Object.keys(reasons);
    var degraded = keys.length > 0;
    $("s-mode").textContent = degraded ? "⚠ DEGRADADO" : "NORMAL";
    $("s-mode-box").className = "item" + (degraded ? " degraded" : "");
    $("s-reasons").textContent = keys.map(function (k) { return reasons[k]; }).join(" · ");
  }

  $("s-day").textContent = index.day;
  $("s-updated").textContent = data.generated_at || index.generated_at;
  document.title = "viz · " + index.day;

  var major = parseInt(String(data.tiles_version).split(".")[0], 10);
  if (!(major <= KNOWN_MAJOR)) {
    setReason("version", "tiles_version " + data.tiles_version + " no soportada: se rechaza el día");
    renderStatus();
    $("panels").textContent = "Versión de tiles no soportada (" + data.tiles_version + ").";
    return;
  }
  if (missingThetas.length) {
    setReason("missing", missingThetas.length + " θ sin datos en L2");
  }
  // Un tile que el índice lista y la página no trae es un hueco visible, no un nivel que se omite en silencio.
  levels.forEach(function (w) {
    ["price", "volume", "dir"].forEach(function (kind) {
      if (kind === "dir" && !thetas.length) return;
      var name = index[kind][w];
      if (files[name] === undefined) setReason("tile:" + name, "falta " + name);
    });
  });

  /* ---------- Decodificación: una pasada por tile ---------- */

  function bytesOf(name) {
    var text = files[name];
    if (text === undefined) {
      setReason("tile:" + name, "falta " + name);
      return null;
    }
    delete files[name]; // el base64 se suelta apenas se decodifica: un solo dato vivo
    var bin = atob(text);
    var out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    metrics.decoded_bytes += out.length;
    return out;
  }

  function bad(name, why) {
    setReason("tile:" + name, name + ": " + why);
    return null;
  }

  function decodePrice(name, w) {
    var bytes = bytesOf(name);
    if (!bytes) return null;
    if (bytes.length !== 32 * w) return bad(name, "tamaño inesperado");
    var n = 4 * w;
    var t = new Uint32Array(bytes.buffer, 0, n); // ms desde el inicio del día
    var p = new Int32Array(bytes.buffer, 16 * w, n); // unidades de 1 / price_scale
    var x = new Float64Array(n);
    var y = new Array(n);
    for (var i = 0; i < n; i++) {
      x[i] = t[i] / 1000;
      y[i] = p[i] === EMPTY ? null : p[i] / scale;
    }
    return { x: x, y: y };
  }

  function decodeVolume(name, w) {
    var bytes = bytesOf(name);
    if (!bytes) return null;
    if (bytes.length !== 4 * w) return bad(name, "tamaño inesperado");
    return new Float32Array(bytes.buffer, 0, w);
  }

  function decodeDir(name, w) {
    var bytes = bytesOf(name);
    if (!bytes) return null;
    if (bytes.length !== thetas.length * w) return bad(name, "tamaño inesperado");
    return bytes;
  }

  var cache = {}; // w -> nivel decodificado

  function hasLevel(w) {
    return cache[w] !== undefined || files[index.price[w]] !== undefined;
  }

  function loadLevel(w) {
    if (cache[w]) return cache[w];
    var t0 = performance.now();
    var price = decodePrice(index.price[w], w);
    if (!price) return null;
    var vol = decodeVolume(index.volume[w], w);
    var dir = thetas.length ? decodeDir(index.dir[w], w) : null;
    var colStart = new Float64Array(w);
    for (var c = 0; c < w; c++) colStart[c] = (c * DAY_S) / w; // posición de la columna
    var level = {
      w: w,
      colDur: DAY_S / w,
      price: price,
      vol: vol,
      dir: dir,
      priceData: [price.x, price.y],
      // Sin tile de volumen no hay barras: nulls, nunca ceros (principio 6).
      volData: [colStart, vol || new Array(w).fill(null)],
    };
    cache[w] = level;
    var ms = performance.now() - t0;
    metrics.decode_ms += ms;
    console.info(
      "viz: nivel " + w + " decodificado en " + ms.toFixed(1) + " ms; total " +
        metrics.decoded_bytes + " B decodificados"
    );
    renderStatus();
    return level;
  }

  /* ---------- Elección de nivel ---------- */

  var panels = $("panels");
  var AXIS_W = 76;

  function plotPx() {
    if (price && price.bbox) return price.bbox.width / (window.devicePixelRatio || 1);
    return Math.max(200, (panels.clientWidth || 1200) - AXIS_W - 12);
  }

  // El nivel más fino que se necesita es el primero cuyas columnas visibles
  // alcanzan los píxeles del gráfico; si ninguno alcanza, el más fino que hay.
  function pickLevel(spanS) {
    var need = (plotPx() * DAY_S) / spanS;
    var last = null;
    for (var i = 0; i < levels.length; i++) {
      var w = levels[i];
      if (!hasLevel(w)) continue;
      last = w;
      if (w >= need) return w;
    }
    return last;
  }

  /* ---------- Selección de θ ---------- */

  var select = $("theta");
  var sel = { kind: "none", k: -1, theta: null }; // kind: "t" (con bloque), "m" (faltante)

  thetas.forEach(function (t, k) {
    var o = document.createElement("option");
    o.value = "t:" + k;
    o.textContent = t.theta + " · " + t.events + " eventos" + (t.provisional_from_s !== null ? " · provisional" : "");
    select.appendChild(o);
  });
  missingThetas.forEach(function (name) {
    var o = document.createElement("option");
    o.value = "m:" + name;
    o.textContent = "⚠ " + name + " · sin datos";
    select.appendChild(o);
  });

  function selectTheta(value) {
    var kind = value.charAt(0);
    var rest = value.slice(2);
    if (kind === "t") sel = { kind: "t", k: parseInt(rest, 10), theta: thetas[parseInt(rest, 10)].theta };
    else sel = { kind: "m", k: -1, theta: rest };
    setReason("theta", kind === "m" ? "θ " + rest + " sin datos: hueco, no se rellena" : null);
    setReason("reserved", null);
    renderStatus();
  }

  if (select.options.length) {
    select.selectedIndex = 0;
    selectTheta(select.value);
  } else {
    setReason("theta", "sin θ en el día");
    renderStatus();
  }

  /* ---------- Formato ---------- */

  function pad2(n) {
    return (n < 10 ? "0" : "") + n;
  }

  function clock(sec, dec) {
    var s = Math.max(0, sec);
    var h = Math.floor(s / 3600);
    var m = Math.floor((s % 3600) / 60);
    var r = s % 60;
    var tail = dec ? (r < 10 ? "0" : "") + r.toFixed(dec) : pad2(Math.floor(r));
    return pad2(h) + ":" + pad2(m) + ":" + tail;
  }

  function fixed(v) {
    return v.toFixed(decimals);
  }

  /* ---------- Dibujo de las regiones (en el lienzo de uPlot, bajo la serie) ---------- */

  var cur = null; // nivel en pantalla
  var price = null;
  var vol = null;
  var thetaT0 = null;
  var hovered = null; // panel bajo el puntero: el único que muestra el tooltip

  function dirBlock(level) {
    if (sel.kind !== "t" || !level.dir) return null;
    return level.dir.subarray(sel.k * level.w, (sel.k + 1) * level.w);
  }

  function emptyColumn(level, c) {
    return level.price.y[4 * c] === null;
  }

  function drawRegions(u) {
    var level = cur;
    var ctx = u.ctx;
    var b = u.bbox;
    var dpr = window.devicePixelRatio || 1;
    var xs = u.scales.x;
    var dir = dirBlock(level);
    var c0 = Math.max(0, Math.floor(xs.min / level.colDur));
    var c1 = Math.min(level.w - 1, Math.ceil(xs.max / level.colDur) - 1);
    var reserved = 0;

    ctx.save();
    ctx.beginPath();
    ctx.rect(b.left, b.top, b.width, b.height);
    ctx.clip();
    ctx.font = 12 * dpr + "px " + MONO;
    ctx.textBaseline = "alphabetic";

    var c = c0;
    while (c <= c1) {
      var empty = emptyColumn(level, c);
      var state = dir ? dir[c] : 0;
      if (state > 4) {
        reserved++;
        state = 0; // reservado: se trata como 0 y se señala
      }
      var end = c;
      while (end + 1 <= c1) {
        var e2 = emptyColumn(level, end + 1);
        var s2 = dir ? dir[end + 1] : 0;
        if (s2 > 4) s2 = 0;
        if (e2 !== empty || s2 !== state) break;
        end++;
      }
      var x0 = u.valToPos(c * level.colDur, "x", true);
      var x1 = u.valToPos((end + 1) * level.colDur, "x", true);
      var wpx = Math.max(1, x1 - x0);

      if (empty) {
        // Hueco: columnas sin ticks. Marcador y texto; nunca se rellena.
        ctx.fillStyle = "rgba(227, 179, 65, 0.10)";
        ctx.fillRect(x0, b.top, wpx, b.height);
        ctx.fillStyle = C.amber;
        ctx.fillRect(x0, b.top + b.height - 3 * dpr, wpx, 3 * dpr);
        if (wpx > 70 * dpr) {
          ctx.textAlign = "center";
          ctx.fillText("◇ sin ticks", x0 + wpx / 2, b.top + b.height - 8 * dpr);
        } else if (wpx > 14 * dpr) {
          ctx.textAlign = "center";
          ctx.fillText("◇", x0 + wpx / 2, b.top + b.height - 8 * dpr);
        }
      } else if (state !== 0) {
        var st = STATES[state];
        var rgb = st.sign > 0 ? C.up : C.down;
        ctx.fillStyle = "rgba(" + rgb + ", " + (st.strong ? 0.42 : 0.18) + ")";
        ctx.fillRect(x0, b.top, wpx, b.height);
        // Forma y posición además del color (principio 2): alza arriba, baja abajo;
        // franja gruesa en el overshoot y fina en la confirmación.
        var band = (st.strong ? 5 : 2) * dpr;
        ctx.fillStyle = "rgb(" + rgb + ")";
        ctx.fillRect(x0, st.sign > 0 ? b.top : b.top + b.height - band, wpx, band);
        if (wpx > 16 * dpr) {
          ctx.fillStyle = C.text; // el glifo va en el color del texto (≥ 7:1); el tono lo da el relleno
          ctx.textAlign = "left";
          ctx.fillText(
            st.sign > 0 ? "▲" : "▼",
            x0 + 3 * dpr,
            st.sign > 0 ? b.top + band + 12 * dpr : b.top + b.height - band - 4 * dpr
          );
        }
      }
      c = end + 1;
    }

    // Cola provisional del θ activo (RVZ-02): marcador con texto, no solo color.
    if (sel.kind === "t" && thetas[sel.k].provisional_from_s !== null) {
      var px = u.valToPos(thetas[sel.k].provisional_from_s, "x", true);
      if (px >= b.left && px <= b.left + b.width) {
        ctx.strokeStyle = C.amber;
        ctx.lineWidth = dpr;
        ctx.setLineDash([4 * dpr, 4 * dpr]);
        ctx.beginPath();
        ctx.moveTo(px, b.top);
        ctx.lineTo(px, b.top + b.height);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.amber;
        ctx.textAlign = "left";
        ctx.fillText("provisional ▸", px + 4 * dpr, b.top + 24 * dpr);
      }
    }
    ctx.restore();

    var text = reserved ? "estado reservado en " + reserved + " columnas (se trata como sin evento)" : null;
    if ((reasons.reserved || null) !== text) {
      setReason("reserved", text);
      renderStatus();
    }
  }

  function drawMessages(u) {
    var msg = null;
    if (sel.kind === "m") msg = "⚠ θ " + sel.theta + ": sin datos en L2 (hueco, no se rellena)";
    else if (sel.kind === "t" && !cur.dir) msg = "⚠ falta el tile de dirección de este nivel";
    if (msg) drawNote(u, msg);
  }

  function drawVolumeMessage(u) {
    if (!cur.vol) drawNote(u, "⚠ falta el tile de volumen de este nivel");
  }

  function drawNote(u, msg) {
    var b = u.bbox;
    var dpr = window.devicePixelRatio || 1;
    var ctx = u.ctx;
    ctx.save();
    ctx.font = "bold " + 14 * dpr + "px " + MONO;
    ctx.textAlign = "center";
    ctx.fillStyle = C.amber;
    ctx.fillText(msg, b.left + b.width / 2, b.top + 22 * dpr);
    ctx.restore();
  }

  /* ---------- Tooltip por cubeta ---------- */

  var tip = $("tip");

  function onCursor(u) {
    var left = u.cursor.left;
    if (left == null || left < 0) {
      tip.hidden = true;
      return;
    }
    if (u !== hovered) return; // el cursor sincronizado: lo atiende el panel con el puntero
    var level = cur;
    var sec = u.posToVal(left, "x");
    var c = Math.min(level.w - 1, Math.max(0, Math.floor(sec / level.colDur)));
    var dec = Number.isInteger(level.colDur) ? 0 : 3;
    var lines = [clock(c * level.colDur, dec) + " – " + clock((c + 1) * level.colDur, dec) + " UTC"];
    if (emptyColumn(level, c)) {
      lines.push("◇ sin ticks en la cubeta (hueco)");
    } else {
      var y = level.price.y;
      var lo = Infinity;
      var hi = -Infinity;
      for (var k = 4 * c; k < 4 * c + 4; k++) {
        if (y[k] === null) continue;
        if (y[k] < lo) lo = y[k];
        if (y[k] > hi) hi = y[k];
      }
      lines.push("mín " + fixed(lo) + "   máx " + fixed(hi));
      lines.push(level.vol ? "vol " + level.vol[c].toFixed(4) : "vol: ⚠ falta el tile");
    }
    if (sel.kind === "t" && !level.dir) {
      lines.push("θ " + sel.theta + ": ⚠ falta el tile de dirección");
    } else if (sel.kind === "t") {
      var code = level.dir[sel.k * level.w + c];
      lines.push("θ " + sel.theta + ": " + (STATES[code] ? STATES[code].text : "reservado (" + code + ")"));
    } else if (sel.kind === "m") {
      lines.push("θ " + sel.theta + ": sin datos");
    }
    tip.textContent = lines.join("\n");
    tip.hidden = false;
    var box = u.over.getBoundingClientRect();
    var x = box.left + left + 14;
    var yy = box.top + u.cursor.top + 14;
    if (x + tip.offsetWidth > window.innerWidth - 4) x = box.left + left - tip.offsetWidth - 14;
    if (yy + tip.offsetHeight > window.innerHeight - 4) yy = box.top + u.cursor.top - tip.offsetHeight - 14;
    tip.style.left = Math.max(0, x) + "px";
    tip.style.top = Math.max(0, yy) + "px";
  }

  /* ---------- Los dos paneles ---------- */

  var syncing = false;
  var levelQueued = false;

  function onScale(u, key) {
    if (key !== "x" || !price || !vol) return;
    var other = u === price ? vol : price;
    var s = u.scales.x;
    var t = other.scales.x;
    if (!syncing && (t.min !== s.min || t.max !== s.max)) {
      syncing = true;
      other.setScale("x", { min: s.min, max: s.max });
      syncing = false;
    }
    queueLevel();
  }

  function queueLevel() {
    if (levelQueued) return;
    levelQueued = true;
    queueMicrotask(ensureLevel);
  }

  function ensureLevel() {
    levelQueued = false;
    var x = price.scales.x;
    var span = x.max - x.min;
    if (!(span > 0)) return;
    var w = pickLevel(span);
    if (w === null || w === cur.w) return;
    var level = loadLevel(w);
    if (!level) return;
    // Al hacer zoom: el nivel más fino que cubre el rango visible, ya en memoria.
    cur = level;
    price.setData(level.priceData, false);
    vol.setData(level.volData, false);
    price.setScale("x", { min: x.min, max: x.max });
    vol.setScale("x", { min: x.min, max: x.max });
    renderFooter();
  }

  function resetZoom() {
    price.setScale("x", { min: 0, max: DAY_S });
  }

  function sizes() {
    var w = panels.clientWidth || 1200;
    var h = panels.clientHeight || 700;
    var hv = Math.round(h * 0.23); // el volumen ocupa 20 a 25 % de la altura
    return { w: w, hp: h - hv, hv: hv };
  }

  function yPrice(u, min, max) {
    if (!isFinite(min) || !isFinite(max)) return [0, 1];
    var pad = (max - min) * 0.05 || 1;
    return [min - pad, max + pad];
  }

  function yVolume(u, min, max) {
    return [0, isFinite(max) && max > 0 ? max * 1.08 : 1];
  }

  function axisBase() {
    return {
      stroke: C.muted,
      font: "12px " + MONO,
      grid: { stroke: C.grid, width: 1 },
      ticks: { stroke: C.grid, width: 1 },
    };
  }

  var X_INCRS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 14400, 21600, 43200];

  function xAxis(labels) {
    var a = axisBase();
    a.scale = "x";
    a.size = labels ? 28 : 6;
    a.space = 90;
    a.incrs = X_INCRS;
    a.values = function (u, splits, axisIdx, space, incr) {
      return splits.map(function (v) {
        if (!labels) return "";
        return incr >= 60 ? clock(v, 0).slice(0, 5) : clock(v, 0);
      });
    };
    return a;
  }

  function yAxis(scaleKey, fmt) {
    var a = axisBase();
    a.scale = scaleKey;
    a.size = AXIS_W;
    a.values = function (u, splits) {
      return splits.map(fmt);
    };
    return a;
  }

  function cursorOpts() {
    return {
      sync: { key: "viz", scales: ["x", null] },
      x: true,
      y: false,
      drag: { x: true, y: false, uni: 20 },
      points: { show: false },
    };
  }

  function build() {
    var s = sizes();
    var shared = { legend: { show: false }, cursor: cursorOpts() };

    price = new uPlot(
      Object.assign({}, shared, {
        width: s.w,
        height: s.hp,
        padding: [8, 12, 0, 0],
        scales: { x: { time: false, auto: false, min: 0, max: DAY_S }, p: { range: yPrice } },
        axes: [xAxis(false), yAxis("p", fixed)],
        series: [{}, { scale: "p", stroke: C.price, width: 1.5, points: { show: false } }],
        hooks: {
          drawClear: [drawRegions],
          draw: [
            function (u) {
              drawMessages(u);
              var now = performance.now();
              if (metrics.first_paint_ms === null) {
                metrics.first_paint_ms = now;
                console.info(
                  "viz: primer trazo a " + now.toFixed(1) + " ms desde el inicio de la navegación (" +
                    (now - started).toFixed(1) + " ms desde que arrancó el script)"
                );
              }
              if (thetaT0 !== null) {
                var dt = now - thetaT0;
                thetaT0 = null;
                metrics.theta_change_ms.push(dt);
                console.info("viz: cambio de θ en " + dt.toFixed(1) + " ms");
              }
            },
          ],
          setCursor: [onCursor],
          setScale: [onScale],
        },
      }),
      cur.priceData,
      $("price")
    );

    vol = new uPlot(
      Object.assign({}, shared, {
        width: s.w,
        height: s.hv,
        padding: [4, 12, 0, 0],
        scales: { x: { time: false, auto: false, min: 0, max: DAY_S }, v: { range: yVolume } },
        axes: [
          xAxis(true),
          yAxis("v", function (v) {
            return v >= 10 ? v.toFixed(0) : v.toFixed(2);
          }),
        ],
        series: [
          {},
          {
            scale: "v",
            stroke: C.volume,
            fill: C.volume,
            width: 0,
            points: { show: false },
            paths: uPlot.paths.bars({ size: [0.9, Infinity, 1], align: 1 }),
          },
        ],
        hooks: { draw: [drawVolumeMessage], setCursor: [onCursor], setScale: [onScale] },
      }),
      cur.volData,
      $("volume")
    );

    [price, vol].forEach(function (u) {
      u.over.addEventListener("dblclick", resetZoom);
      u.over.addEventListener("mouseenter", function () { hovered = u; });
      u.over.addEventListener("mouseleave", function () { hovered = null; tip.hidden = true; });
    });
  }

  function renderFooter() {
    var dur = cur.colDur;
    $("f-level").textContent =
      "nivel " + cur.w + " · " + (Number.isInteger(dur) ? dur : dur.toFixed(2)) + " s por columna";
  }

  /* ---------- Arranque ---------- */

  var first = pickLevel(DAY_S);
  cur = first === null ? null : loadLevel(first);
  if (!cur) {
    setReason("tiles", "no hay ningún nivel de precio utilizable");
    renderStatus();
    panels.textContent = "Sin tiles de precio: el día no se puede dibujar.";
    return;
  }
  renderStatus();
  renderFooter();
  build();

  select.addEventListener("change", function () {
    thetaT0 = performance.now();
    selectTheta(select.value);
    price.redraw(false); // solo repinta: el bloque de θ ya está en memoria
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") resetZoom();
  });

  window.addEventListener("resize", function () {
    var s = sizes();
    price.setSize({ width: s.w, height: s.hp });
    vol.setSize({ width: s.w, height: s.hv });
    queueLevel();
  });
})();
