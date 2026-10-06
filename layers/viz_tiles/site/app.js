/* Vista de un día (TRD-viz §6.6, §6.7, §6.11 a §6.14, §7). Sin red: los dos archivos del día llegan en
 * window.VIZ_DATA, en base64 bajo su nombre. `ticks.bin` (§7.3) son todos los ticks del día en tramos
 * de tres secciones de varint; `events.bin` (§7.4), los eventos exactos de cada θ. El navegador los decodifica
 * una sola vez a arreglos tipados y, en cada dibujo, deriva de ellos lo que se ve: por cada píxel de
 * ancho de la escala actual, el precio (un punto por tick, o el segmento del mínimo al máximo), el volumen
 * (suma de la cantidad) y las confirmaciones (θ que confirman, y máximo en el mismo instante). Lo que se
 * dibuja es un tick o la envolvente exacta de los ticks de un píxel, a cualquier zoom (ADR-VZ-14).
 * Sin telemetría: las métricas van a la consola. */
(function () {
  "use strict";

  var DAY_S = 86400;
  var KNOWN_MAJOR = 2;
  var QTY_SCALE = 1e8; // la cantidad de ticks.bin está en unidades de 10⁻⁸ (§7.3)
  var DOT_PX = 3; // lado de un punto de precio, en px CSS
  var MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace';
  var C = {
    muted: "#9aa7b4",
    mutedLine: "rgba(154, 167, 180, 0.6)",
    grid: "rgba(154, 167, 180, 0.16)",
    price: "#e6edf3",
    volume: "rgba(110, 138, 168, 0.85)",
    confirms: "rgba(86, 182, 194, 0.42)",
    simul: "rgb(86, 182, 194)",
    amber: "#e3b341",
    up: "63, 185, 80",
    down: "248, 81, 73",
  };
  // Banderas de un evento (§7.4).
  var F_UP = 1;
  var F_PROVISIONAL = 2;
  var F_REF_CLIPPED = 4;
  var F_CONFIRM_CLIPPED = 8;
  var F_EXTREME_CLIPPED = 16;
  var EVENT_BYTES = 13;
  // Franja del borde, en px CSS: fina en la confirmación, gruesa en el overshoot.
  var BAND_PX = { thin: 3, thick: 8 };
  // Eventos enteros (de la referencia al extremo) más angostos que estos px CSS que caben juntos en
  // ese ancho se cuentan en una sola marca de densidad.
  var DENSE_PX = 3;
  // Margen a cada lado de "ajustar a la ventana", como fracción de la ventana.
  var FIT_MARGIN = 0.05;
  // Reparto vertical (§6.11): precio, confirmaciones, volumen.
  var SPLIT = { price: 0.65, confirms: 0.15, volume: 0.2 };
  var TIP_MAX_LINES = 12;

  var started = performance.now();
  var metrics = (window.VIZ_METRICS = {
    decoded_bytes: 0,
    decode_ms: 0,
    ticks: 0,
    first_paint_ms: null,
    theta_change_ms: [],
    draw_ms: 0,
    frame: null,
  });

  function $(id) {
    return document.getElementById(id);
  }

  var data = window.VIZ_DATA;
  var dataScript = $("viz-data");
  if (dataScript) dataScript.remove(); // el texto base64 deja de vivir también en el DOM

  if (!data || !data.index) {
    $("panels").textContent = "Sin datos: la página no trae el día.";
    return;
  }

  var index = data.index;
  var files = data.files;
  var scale = index.price_scale;
  var decimals = Math.round(Math.log10(scale));
  var thetas = index.thetas;
  var missingThetas = index.missing_thetas;

  /* ---------- Datos completos o incompletos (principio 6) ---------- */

  var reasons = {}; // clave -> texto; si hay alguna, los datos están incompletos

  function setReason(key, text) {
    if (text) reasons[key] = text;
    else delete reasons[key];
  }

  function renderStatus() {
    var keys = Object.keys(reasons);
    var incomplete = keys.length > 0;
    $("s-data").textContent = incomplete
      ? "incompletos: " + keys.map(function (k) { return reasons[k]; }).join(" · ")
      : "completos";
    $("s-data-box").className = "item" + (incomplete ? " incomplete" : "");
  }

  $("s-day").textContent = index.day;
  $("s-updated").textContent = data.generated_at || index.generated_at;
  document.title = "viz · " + index.day;

  var major = parseInt(String(data.tiles_version).split(".")[0], 10);
  if (major !== KNOWN_MAJOR) {
    setReason("version", "tiles_version " + data.tiles_version + " no soportada: se rechaza el día");
    renderStatus();
    $("panels").textContent = "Versión de tiles no soportada (" + data.tiles_version + ").";
    return;
  }
  if (missingThetas.length) {
    setReason("missing", missingThetas.length + " θ sin datos en L2");
  }

  /* ---------- Decodificación: una sola vez, al abrir ---------- */

  function bytesOf(name) {
    var text = files[name];
    if (text === undefined) {
      setReason("file:" + name, "falta " + name);
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
    setReason("file:" + name, name + ": " + why);
    return null;
  }

  // Los ticks (§7.3): una secuencia de tramos de hasta `ticks_chunk` ticks. Cada tramo trae una cabecera
  // de cuatro uint32 little-endian (ticks y bytes del Δtiempo, del Δprecio y de la cantidad) y sus tres
  // secciones de enteros varint (LEB128). El Δtiempo y el Δprecio del primer tick de un tramo son relativos
  // al último del anterior: los acumuladores siguen de un tramo al otro. Tiempo en ms desde el inicio del
  // día (Int32Array), precio en unidades de 1 / price_scale (Int32Array) y cantidad (Float64Array).
  var ticks = null; // { n, t, p, q }
  var CHUNK_HEADER = 16;

  function decodeTicks(name, n) {
    var bytes = bytesOf(name);
    if (!bytes) return null;
    var len = bytes.length;
    var view = new DataView(bytes.buffer, bytes.byteOffset, len);
    var maxChunk = index.ticks_chunk;
    var pos = 0;
    var end = 0; // fin de la sección en curso
    var overrun = false;
    // Aritmética de coma flotante y no operadores de bits: la cantidad pasa de 2³² (1 000 BTC).
    function varint() {
      if (pos >= end) {
        overrun = true;
        return 0;
      }
      var b = bytes[pos++];
      if (b < 128) return b;
      var r = b & 127;
      var m = 128;
      do {
        if (pos >= end) {
          overrun = true;
          return 0;
        }
        b = bytes[pos++];
        r += (b & 127) * m;
        m *= 128;
      } while (b >= 128);
      return r;
    }
    var t = new Int32Array(n);
    var p = new Int32Array(n);
    var q = new Float64Array(n);
    var at = 0; // ticks ya decodificados
    var accT = 0;
    var accP = 0;
    var i;
    while (pos < len) {
      if (len - pos < CHUNK_HEADER) return bad(name, "tamaño inesperado (cabecera de tramo truncada)");
      var count = view.getUint32(pos, true);
      var size = [view.getUint32(pos + 4, true), view.getUint32(pos + 8, true), view.getUint32(pos + 12, true)];
      pos += CHUNK_HEADER;
      if (count === 0 || count > maxChunk || at + count > n) return bad(name, "tamaño inesperado (tramo fuera de lo declarado)");
      if (size[0] + size[1] + size[2] > len - pos) return bad(name, "tamaño inesperado (tramo truncado)");
      var stop = at + count;
      end = pos + size[0];
      for (i = at; i < stop; i++) {
        accT += varint();
        t[i] = accT;
      }
      if (overrun || pos !== end) return bad(name, "tamaño inesperado (sección de tramo)");
      end = pos + size[1];
      for (i = at; i < stop; i++) {
        var z = varint();
        accP += z % 2 === 0 ? z / 2 : -(z + 1) / 2; // zigzag
        p[i] = accP;
      }
      if (overrun || pos !== end) return bad(name, "tamaño inesperado (sección de tramo)");
      end = pos + size[2];
      for (i = at; i < stop; i++) q[i] = varint() / QTY_SCALE;
      if (overrun || pos !== end) return bad(name, "tamaño inesperado (sección de tramo)");
      at = stop;
    }
    if (at !== n) return bad(name, "tamaño inesperado");
    return { n: n, t: t, p: p, q: q };
  }

  function loadTicks() {
    var t0 = performance.now();
    ticks = decodeTicks(index.ticks_file, index.ticks);
    var ms = performance.now() - t0;
    metrics.decode_ms += ms;
    if (ticks) {
      metrics.ticks = ticks.n;
      console.info(
        "viz: ticks decodificados en " + ms.toFixed(1) + " ms (" + ticks.n + " ticks, " +
          metrics.decoded_bytes + " B decodificados)"
      );
    }
    renderStatus();
  }

  // Eventos exactos de todos los θ del día (§7.4): cuatro secciones de N valores. El θ k ocupa
  // de events_offset a events_offset + events - 1 en cada una.
  var ev = null;
  // Las confirmaciones de todos los θ en orden de hora: confT (ms) y confK (índice del θ).
  var conf = null;

  function loadEvents() {
    var name = index.events;
    if (!name) {
      setReason("events", "el índice no lista los eventos exactos");
      return;
    }
    var bytes = bytesOf(name);
    if (!bytes) return;
    var total = 0;
    thetas.forEach(function (t) { total += t.events; });
    if (bytes.length !== EVENT_BYTES * total) {
      bad(name, "tamaño inesperado");
      return;
    }
    ev = {
      n: total,
      ref: new Int32Array(bytes.buffer, 0, total), // ms desde el inicio del día
      conf: new Int32Array(bytes.buffer, 4 * total, total),
      ext: new Int32Array(bytes.buffer, 8 * total, total),
      flags: new Uint8Array(bytes.buffer, 12 * total, total),
    };
    buildConfirmations();
  }

  // Las confirmaciones de los θ juntas y ordenadas por hora: la clave `hora * 256 + θ` cabe en un
  // Float64 exacto y se ordena con el sort numérico nativo, sin comparador.
  function buildConfirmations() {
    var keys = [];
    thetas.forEach(function (t, k) {
      for (var i = 0; i < t.events; i++) {
        var e = t.events_offset + i;
        if (!(ev.flags[e] & F_CONFIRM_CLIPPED)) keys.push(ev.conf[e] * 256 + k);
      }
    });
    var sorted = Float64Array.from(keys).sort();
    var n = sorted.length;
    conf = { n: n, t: new Int32Array(n), k: new Uint8Array(n) };
    for (var j = 0; j < n; j++) {
      conf.k[j] = sorted[j] % 256;
      conf.t[j] = (sorted[j] - conf.k[j]) / 256;
    }
  }

  /* ---------- Búsquedas ---------- */

  // Primer índice con arr[i] >= value (arr no decreciente).
  function lowerBoundAll(arr, n, value) {
    var lo = 0;
    var hi = n;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (arr[mid] < value) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  // El bloque del θ k en `ev`: {off, n}, o null si no hay θ con bloque o faltan los eventos.
  function block(k) {
    if (!ev || k < 0 || !thetas[k]) return null;
    return { off: thetas[k].events_offset, n: thetas[k].events };
  }

  // Primer evento i del bloque con arr[off + i] >= value (los tres tiempos crecen con i).
  function lowerBound(arr, b, value) {
    var lo = 0;
    var hi = b.n;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (arr[b.off + mid] < value) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  /* ---------- Selección de θ ---------- */

  var select = $("theta");
  var sel = { kind: "none", k: -1, theta: null }; // kind: "t" (con eventos), "m" (faltante)

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

  function activeBlock() {
    return sel.kind === "t" ? block(sel.k) : null;
  }

  function selectTheta(value) {
    var kind = value.charAt(0);
    var rest = value.slice(2);
    if (kind === "t") sel = { kind: "t", k: parseInt(rest, 10), theta: thetas[parseInt(rest, 10)].theta };
    else sel = { kind: "m", k: -1, theta: rest };
    setReason("theta", kind === "m" ? "θ " + rest + " sin datos: hueco, no se rellena" : null);
    nav.k = -1; // los índices de evento son de cada θ
    renderNav(null);
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

  // Hora de un instante en ms desde el inicio del día, con enteros: sin errores de redondeo en los ms.
  function clockMs(ms, dec) {
    var v = Math.max(0, Math.round(ms));
    var h = Math.floor(v / 3600000);
    var m = Math.floor((v % 3600000) / 60000);
    var s = Math.floor((v % 60000) / 1000);
    var f = v % 1000;
    var out = pad2(h) + ":" + pad2(m) + ":" + pad2(s);
    if (dec) out += "." + (f < 100 ? "0" : "") + (f < 10 ? "0" : "") + f;
    return out;
  }

  function fixed(v) {
    return v.toFixed(decimals);
  }

  function plural(n, one, many) {
    return n + " " + (n === 1 ? one : many);
  }

  /* ---------- Lo que se ve: el marco de un dibujo ---------- */

  // Un marco son los números por píxel de la escala x actual, calculados en un recorrido lineal sobre
  // los ticks visibles (con búsqueda binaria para el primero) y sobre las confirmaciones visibles. Lo
  // comparten los tres paneles: precio, volumen y confirmaciones salen del mismo recorrido.
  //  - cnt, lo, hi, vol: ticks, precio mínimo y máximo (unidades de 1/price_scale) y cantidad total de
  //    los ticks del píxel;
  //  - cc, cs: θ que confirman en el píxel y máximo de θ que confirman en el mismo instante.
  // El píxel de un tick es una función de su ms: los ticks de un mismo instante nunca se separan.
  var frame = null;

  function plotPx(u) {
    var dpr = window.devicePixelRatio || 1;
    if (u && u.bbox && u.bbox.width) return Math.max(1, Math.round(u.bbox.width / dpr));
    return Math.max(200, (panels.clientWidth || 1200) - AXIS_W - 12);
  }

  function getFrame(u) {
    var xs = u.scales.x;
    var lo = isFinite(xs.min) ? xs.min : 0;
    var hi = isFinite(xs.max) ? xs.max : DAY_S;
    var cols = plotPx(u);
    if (frame && frame.xmin === lo && frame.xmax === hi && frame.cols === cols) return frame;
    frame = computeFrame(lo, hi, cols);
    return frame;
  }

  function computeFrame(xmin, xmax, cols) {
    var t0 = performance.now();
    var a = xmin * 1000;
    var z = xmax * 1000;
    var k = cols / Math.max(z - a, 1e-6); // px CSS por ms
    var f = {
      xmin: xmin,
      xmax: xmax,
      cols: cols,
      a: a,
      z: z,
      k: k,
      cnt: new Uint32Array(cols),
      lo: new Int32Array(cols),
      hi: new Int32Array(cols),
      vol: new Float64Array(cols),
      cc: new Uint16Array(cols),
      cs: new Uint16Array(cols),
      n: 0,
      pMin: 0,
      pMax: 0,
      volMax: 0,
      ccMax: 0,
      ms: 0,
    };
    if (ticks) {
      var T = ticks.t;
      var P = ticks.p;
      var Q = ticks.q;
      var i = lowerBoundAll(T, ticks.n, Math.ceil(a));
      var first = i;
      var pMin = Infinity;
      var pMax = -Infinity;
      for (; i < ticks.n; i++) {
        var t = T[i];
        if (t > z) break;
        var px = ((t - a) * k) | 0; // no negativo y menor que 2³¹: igual que floor, sin la búsqueda global de Math
        if (px >= cols) px = cols - 1;
        var p = P[i];
        if (f.cnt[px] === 0) {
          f.lo[px] = p;
          f.hi[px] = p;
        } else {
          if (p < f.lo[px]) f.lo[px] = p;
          if (p > f.hi[px]) f.hi[px] = p;
        }
        f.cnt[px]++;
        f.vol[px] += Q[i];
        if (p < pMin) pMin = p;
        if (p > pMax) pMax = p;
      }
      f.n = i - first;
      if (f.n === 0) {
        // Sin ticks en la vista: el último precio conocido (o el primero que sigue) centra el eje.
        var near = first > 0 ? P[first - 1] : first < ticks.n ? P[first] : 0;
        pMin = near;
        pMax = near;
      }
      f.pMin = pMin;
      f.pMax = pMax;
      for (var c = 0; c < cols; c++) {
        if (f.vol[c] > f.volMax) f.volMax = f.vol[c];
      }
    }
    if (conf) {
      var stampPx = new Int32Array(thetas.length); // último píxel (+1) en que contó cada θ
      var stampT = new Float64Array(thetas.length); // último instante (+1) en que contó cada θ
      var lastT = -1;
      var run = 0;
      for (var j = lowerBoundAll(conf.t, conf.n, Math.ceil(a)); j < conf.n; j++) {
        var ct = conf.t[j];
        if (ct > z) break;
        var cx = ((ct - a) * k) | 0;
        if (cx >= cols) cx = cols - 1;
        var th = conf.k[j];
        if (stampPx[th] !== cx + 1) {
          stampPx[th] = cx + 1;
          f.cc[cx]++;
        }
        if (ct !== lastT) {
          lastT = ct;
          run = 0;
        }
        if (stampT[th] !== ct + 1) {
          stampT[th] = ct + 1;
          run++;
          if (run > f.cs[cx]) f.cs[cx] = run;
        }
      }
      for (var d = 0; d < cols; d++) {
        if (f.cc[d] > f.ccMax) f.ccMax = f.cc[d];
      }
    }
    f.ms = performance.now() - t0;
    metrics.frame = { cols: cols, ticks: f.n, ms: f.ms, pixels_with_ticks: countFilled(f) };
    return f;
  }

  function countFilled(f) {
    var n = 0;
    for (var c = 0; c < f.cols; c++) if (f.cnt[c]) n++;
    return n;
  }

  /* ---------- Dibujo de las regiones (en el lienzo de uPlot, bajo las marcas) ---------- */

  function vline(ctx, x, b, dpr, style) {
    ctx.fillStyle = style;
    ctx.fillRect(Math.round(x - dpr / 2), b.top, dpr, b.height);
  }

  // Una franja de un evento: relleno del tramo y franja del borde (arriba el alza, abajo la baja).
  function band(ctx, b, dpr, x0, x1, up, strong) {
    var rgb = up ? C.up : C.down;
    var w = Math.max(1, x1 - x0);
    ctx.fillStyle = "rgba(" + rgb + ", " + (strong ? 0.42 : 0.18) + ")";
    ctx.fillRect(x0, b.top, w, b.height);
    // Forma y posición además del color (principio 2): franja gruesa en el overshoot y fina en
    // la confirmación. Es la única marca de dirección y fase: no se repite con un glifo.
    var h = (strong ? BAND_PX.thick : BAND_PX.thin) * dpr;
    ctx.fillStyle = "rgb(" + rgb + ")";
    ctx.fillRect(x0, up ? b.top : b.top + b.height - h, w, h);
  }

  // Las franjas de los eventos exactos del θ activo, en sus instantes reales a cualquier zoom, y
  // la marca de densidad donde varios eventos enteros caben en un mismo píxel. Devuelve las marcas.
  function drawEvents(u, ctx, b, dpr) {
    var tr = activeBlock();
    var out = { events: 0, groups: 0, grouped: 0, max: 0 };
    if (!tr || !tr.n) return out;
    var xs = u.scales.x;
    var minW = DENSE_PX * dpr;
    var i = lowerBound(ev.ext, tr, Math.floor(xs.min * 1000)); // primer evento que termina en la vista
    var vis = [];
    var groups = [];
    var g = null;
    for (; i < tr.n; i++) {
      var e = tr.off + i;
      if (ev.ref[e] / 1000 > xs.max) break;
      var v = {
        e: e,
        i: i,
        x0: u.valToPos(ev.ref[e] / 1000, "x", true),
        xc: u.valToPos(ev.conf[e] / 1000, "x", true),
        x1: u.valToPos(ev.ext[e] / 1000, "x", true),
        g: -1,
      };
      v.compact = v.x1 - v.x0 < minW;
      if (!v.compact) {
        g = null; // un evento ancho se ve por sí solo y corta el grupo
      } else if (g && v.x1 - g.anchor < minW) {
        // Eventos enteros dentro de un mismo píxel (cada uno más angosto que DENSE_PX y el extremo a
        // menos de DENSE_PX del extremo del primero): no cuenta el que solo termina aquí.
        g.n++;
        g.last = v.x1;
        v.g = groups.length - 1;
      } else {
        g = { anchor: v.x1, last: v.x1, n: 1 };
        groups.push(g);
        v.g = groups.length - 1;
      }
      vis.push(v);
    }
    out.events = vis.length;

    // Franjas. Un evento entero más angosto que DENSE_PX dentro de un grupo de varios no se
    // dibuja solo: lo cuenta la marca. Uno solo, angosto, se ensancha hasta DENSE_PX para verse.
    vis.forEach(function (v) {
      var f = ev.flags[v.e];
      var up = (f & F_UP) !== 0;
      if (v.compact && groups[v.g].n > 1) return;
      var x1 = v.compact ? v.x0 + minW : v.x1;
      var xc = Math.min(v.xc, x1);
      band(ctx, b, dpr, v.x0, xc, up, false);
      band(ctx, b, dpr, xc, x1, up, true);
      // Confirmación: línea del color del evento. El extremo no lleva línea: el cambio de color
      // entre franjas ya lo marca, porque los eventos alternan siempre. No se dibuja si el tiempo
      // quedó recortado al borde del día.
      if (!(f & F_CONFIRM_CLIPPED)) vline(ctx, xc, b, dpr, "rgb(" + (up ? C.up : C.down) + ")");
    });

    // Marca de densidad: un rectángulo neutro con el número, en vez de franjas indistinguibles.
    var row = 0;
    ctx.font = 12 * dpr + "px " + MONO;
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
    groups.forEach(function (grp) {
      if (grp.n < 2) return;
      var left = grp.anchor - dpr;
      var width = Math.max(minW, grp.last - grp.anchor + 2 * dpr);
      ctx.fillStyle = "rgba(215, 222, 230, 0.26)";
      ctx.fillRect(left, b.top, width, b.height);
      ctx.fillStyle = C.muted;
      ctx.fillText(grp.n + " eventos", left + width + 4 * dpr, b.top + (22 + 14 * (row % 3)) * dpr);
      row++;
      out.groups++;
      out.grouped += grp.n;
      if (grp.n > out.max) out.max = grp.n;
    });

    // El evento elegido con la navegación: un marco sobre su intervalo, tenue (el precio es el
    // único trazo claro del panel).
    if (nav.k >= 0 && nav.k < tr.n) {
      var s = tr.off + nav.k;
      var sx0 = u.valToPos(ev.ref[s] / 1000, "x", true);
      var sx1 = u.valToPos(ev.ext[s] / 1000, "x", true);
      ctx.strokeStyle = C.mutedLine;
      ctx.lineWidth = 2 * dpr;
      ctx.strokeRect(sx0, b.top + dpr, Math.max(minW, sx1 - sx0), b.height - 2 * dpr);
    }
    return out;
  }

  function drawRegions(u) {
    var ctx = u.ctx;
    var b = u.bbox;
    var dpr = window.devicePixelRatio || 1;

    ctx.save();
    ctx.beginPath();
    ctx.rect(b.left, b.top, b.width, b.height);
    ctx.clip();
    ctx.font = 12 * dpr + "px " + MONO;
    ctx.textBaseline = "alphabetic";

    metrics.density = drawEvents(u, ctx, b, dpr);

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
        ctx.fillText("provisional ▸", px + 4 * dpr, b.top + 8 * dpr + 12 * dpr);
      }
    }
    ctx.restore();
  }

  /* ---------- Las marcas por píxel ---------- */

  // Posición vertical (px del lienzo) de un valor en la escala `key` del panel.
  function yPos(u, key, v) {
    var s = u.scales[key];
    var b = u.bbox;
    return b.top + ((s.max - v) / (s.max - s.min)) * b.height;
  }

  // Precio: por cada píxel de ancho, un punto por tick (con 1 o 2 ticks) o el segmento del mínimo al
  // máximo (con 3 o más) y nada más. Nada une un píxel con el vecino: nadie midió lo que hay entre
  // ellos (principio 6). Los ticks de un mismo instante comparten píxel: un solo segmento.
  function drawPrice(u) {
    var f = getFrame(u);
    var ctx = u.ctx;
    var b = u.bbox;
    var dpr = window.devicePixelRatio || 1;
    var colW = b.width / f.cols;
    var half = (DOT_PX * dpr) / 2;
    var line = Math.max(1, Math.round(dpr)); // ancho del segmento: 1 px CSS
    var dots = 0;
    var segments = 0;
    ctx.save();
    ctx.beginPath();
    ctx.rect(b.left, b.top, b.width, b.height);
    ctx.clip();
    ctx.fillStyle = C.price;
    for (var c = 0; c < f.cols; c++) {
      var n = f.cnt[c];
      if (!n) continue;
      var x = b.left + (c + 0.5) * colW;
      var yLo = yPos(u, "p", f.lo[c] / scale); // el precio menor queda más abajo
      var yHi = yPos(u, "p", f.hi[c] / scale);
      if (n <= 2) {
        ctx.fillRect(Math.round(x - half), Math.round(yLo - half), 2 * half, 2 * half);
        dots++;
        if (n === 2 && f.lo[c] !== f.hi[c]) {
          ctx.fillRect(Math.round(x - half), Math.round(yHi - half), 2 * half, 2 * half);
          dots++;
        }
      } else {
        // Al menos un píxel de alto: un rango no se vuelve invisible.
        ctx.fillRect(Math.round(x - line / 2), Math.round(yHi), line, Math.max(line, Math.round(yLo - yHi)));
        segments++;
      }
    }
    ctx.restore();
    metrics.price_draw = { dots: dots, segments: segments };
    renderFooter(f);
  }

  // Barras por píxel (volumen y θ que confirman): una barra por píxel con ticks y, con zoom
  // suficiente, una por tick; un instante compartido suma.
  function drawBars(u, key, values, max, color, f) {
    var ctx = u.ctx;
    var b = u.bbox;
    var dpr = window.devicePixelRatio || 1;
    var colW = b.width / f.cols;
    // A zoom fuerte un instante ocupa varios píxeles: la barra se ensancha sin juntarse con la vecina.
    var width = Math.max(1, Math.min(4, Math.floor(f.k * 0.6))) * dpr;
    var base = b.top + b.height;
    ctx.save();
    ctx.beginPath();
    ctx.rect(b.left, b.top, b.width, b.height);
    ctx.clip();
    ctx.fillStyle = color;
    for (var c = 0; c < f.cols; c++) {
      var v = values[c];
      if (!v) continue;
      var y = yPos(u, key, v);
      var x = b.left + (c + 0.5) * colW;
      ctx.fillRect(Math.round(x - width / 2), Math.round(y), Math.max(1, Math.round(width)), Math.max(1, Math.round(base - y)));
    }
    ctx.restore();
  }

  // Rótulo corto arriba a la izquierda de un panel inferior.
  function drawLabel(u, text) {
    var b = u.bbox;
    var dpr = window.devicePixelRatio || 1;
    var ctx = u.ctx;
    ctx.save();
    ctx.font = 12 * dpr + "px " + MONO;
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = C.muted;
    ctx.fillText(text, b.left + 6 * dpr, b.top + 14 * dpr);
    ctx.restore();
  }

  function drawVolume(u) {
    if (stale(u)) return;
    var f = getFrame(u);
    drawBars(u, "v", f.vol, f.volMax, C.volume, f);
    drawLabel(u, "Volumen");
    if (!ticks) drawNote(u, "⚠ faltan los ticks del día");
  }

  function drawConfirms(u) {
    if (stale(u)) return;
    var f = getFrame(u);
    drawBars(u, "c", f.cc, f.ccMax, C.confirms, f);
    drawBars(u, "c", f.cs, f.ccMax, C.simul, f);
    drawLabel(u, "θ que confirman");
    if (!ev) drawNote(u, "⚠ faltan los eventos exactos");
  }

  function drawMessages(u) {
    var msg = null;
    if (sel.kind === "m") msg = "⚠ θ " + sel.theta + ": sin datos en L2 (hueco, no se rellena)";
    else if (sel.kind === "t" && !ev) msg = "⚠ faltan los eventos exactos: no se dibujan franjas";
    if (msg) drawNote(u, msg);
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

  /* ---------- Tooltip por píxel ---------- */

  var tip = $("tip");

  // El píxel `c` del marco y su rango de ticks [i0, i1) y de ms [a, z).
  function pixelRange(f, c) {
    var a = f.a + c / f.k;
    var z = c === f.cols - 1 ? f.z : f.a + (c + 1) / f.k;
    return { a: a, z: z };
  }

  function pixelLines(f, c) {
    var r = pixelRange(f, c);
    var dec = 1 / f.k >= 1000 ? 0 : 3; // ms, la resolución de los datos; sin fracción desde 1 s por píxel
    var lines = [clockMs(r.a, dec) + " – " + clockMs(r.z, dec) + " UTC"];
    var n = f.cnt[c];
    if (!n) {
      lines.push("sin ticks");
      return lines;
    }
    var i0 = lowerBoundAll(ticks.t, ticks.n, Math.ceil(r.a));
    var i1 = c === f.cols - 1 ? lowerBoundAll(ticks.t, ticks.n, Math.floor(r.z) + 1) : lowerBoundAll(ticks.t, ticks.n, Math.ceil(r.z));
    // Un ms es la cubeta mínima: si todos comparten instante, el rótulo lo dice; si el píxel abarca
    // más de 1 ms, "este ms" no tiene referente y se nombra el instante.
    var sameMs = ticks.t[i0] === ticks.t[i1 - 1];
    var at = "";
    if (sameMs) at = r.z - r.a > 1 ? " a las " + clockMs(ticks.t[i0], 3) : n >= 2 ? " en este ms" : "";
    lines.push(n + " " + (n === 1 ? "tick" : "ticks") + at);
    if (f.lo[c] === f.hi[c]) lines.push("precio " + fixed(f.lo[c] / scale));
    else lines.push("mín " + fixed(f.lo[c] / scale) + "   máx " + fixed(f.hi[c] / scale));
    lines.push("vol " + f.vol[c].toFixed(4));
    return lines;
  }

  // El evento del θ activo que contiene el instante `ms`, o -1: (referencia, extremo]; el tick extremo
  // pertenece al evento que cierra (ADR-VZ-12).
  function eventAt(ms) {
    var tr = activeBlock();
    if (!tr || !tr.n) return -1;
    var i = lowerBound(ev.ref, tr, ms) - 1; // último evento con referencia < ms
    return i >= 0 && ms <= ev.ext[tr.off + i] ? i : -1;
  }

  // Todo lo que se sabe de un evento, para el tooltip: la franja de estado no lo repite.
  function eventLines(i) {
    var tr = activeBlock();
    var e = tr.off + i;
    var f = ev.flags[e];
    var w = windowOf(i);
    var lines = [
      "θ " + sel.theta + " · evento " + (i + 1) + " / " + tr.n + " · " + ((f & F_UP) !== 0 ? "alza" : "baja"),
      "referencia " + clockMs(ev.ref[e], 3),
      "confirmación " + clockMs(ev.conf[e], 3),
      "extremo " + clockMs(ev.ext[e], 3),
      "ventana " + clock(w.start, 3) + " a " + clock(w.end, 3),
    ];
    return lines.concat(w.notes);
  }

  function thetaLines(f, c) {
    var lines = [];
    if (sel.kind === "m") {
      lines.push("θ " + sel.theta + ": sin datos");
    } else if (sel.kind === "t" && ev) {
      var r = pixelRange(f, c);
      var i = eventAt(Math.floor((r.a + r.z) / 2));
      if (i >= 0) lines = lines.concat(eventLines(i));
      else lines.push("θ " + sel.theta + ": fuera de un evento");
    }
    return lines;
  }

  function confirmLines(f, c) {
    var r = pixelRange(f, c);
    var dec = 1 / f.k >= 1000 ? 0 : 3;
    var lines = [clockMs(r.a, dec) + " – " + clockMs(r.z, dec) + " UTC"];
    if (!conf) {
      lines.push("θ que confirman: ⚠ faltan los eventos");
      return lines;
    }
    lines.push("θ que confirman: " + f.cc[c]);
    lines.push("máx. en el mismo instante: " + f.cs[c]);
    var j = lowerBoundAll(conf.t, conf.n, Math.ceil(r.a));
    var end = c === f.cols - 1 ? Math.floor(r.z) + 1 : Math.ceil(r.z);
    var shown = 0;
    var total = 0;
    for (; j < conf.n && conf.t[j] < end; j++) {
      total++;
      if (shown < TIP_MAX_LINES) {
        lines.push("θ " + thetas[conf.k[j]].theta + "  " + clockMs(conf.t[j], 3));
        shown++;
      }
    }
    if (total > shown) lines.push("… y " + (total - shown) + " más");
    return lines;
  }

  function onCursor(u) {
    var left = u.cursor.left;
    if (left == null || left < 0 || !ticks) {
      tip.hidden = true;
      return;
    }
    if (u !== hovered) return; // el cursor sincronizado: lo atiende el panel con el puntero
    var f = getFrame(u);
    var css = u.bbox.width / (window.devicePixelRatio || 1);
    var c = Math.min(f.cols - 1, Math.max(0, Math.floor((left * f.cols) / css)));
    var lines = u === confP ? confirmLines(f, c) : pixelLines(f, c).concat(thetaLines(f, c));
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

  /* ---------- Navegación por eventos ---------- */

  var nav = { k: -1, window: null };
  var prevBtn = $("ev-prev");
  var nextBtn = $("ev-next");
  var fitBtn = $("ev-fit");
  var navInfo = $("ev-info");

  function setX(min, max) {
    pricePlot.setScale("x", { min: min, max: max });
  }

  // La ventana del evento k: de la referencia del anterior al extremo del siguiente, para ver las
  // dos transiciones enteras. Sin vecino dentro del día, el borde es el del propio evento.
  function windowOf(k) {
    var tr = activeBlock();
    var a = Math.max(0, k - 1);
    var z = Math.min(tr.n - 1, k + 1);
    var notes = [];
    if (k === 0) notes.push("sin evento anterior en el día");
    if (k === tr.n - 1) notes.push("sin evento siguiente en el día");
    var clipped = (ev.flags[tr.off + a] & F_REF_CLIPPED) || (ev.flags[tr.off + z] & F_EXTREME_CLIPPED);
    if (clipped) notes.push("la ventana excede el día: recortada al borde");
    if (ev.flags[tr.off + z] & F_PROVISIONAL) {
      notes.push(
        z === k
          ? "el extremo es provisional (candidato vigente)"
          : "el extremo del evento siguiente es provisional (candidato vigente)"
      );
    } else if (ev.flags[tr.off + k] & F_PROVISIONAL) {
      notes.push("el extremo es provisional (candidato vigente)");
    }
    return {
      start: ev.ref[tr.off + a] / 1000,
      end: ev.ext[tr.off + z] / 1000,
      notes: notes,
    };
  }

  // La franja de estado solo dice dónde se está: el detalle del evento va al tooltip.
  function renderNav(w) {
    var tr = activeBlock();
    var has = !!tr && tr.n > 0;
    prevBtn.disabled = !has || nav.k === 0;
    nextBtn.disabled = !has || nav.k === tr.n - 1;
    fitBtn.disabled = !has || nav.k < 0;
    nav.window = w;
    if (!tr) {
      navInfo.textContent = sel.kind === "t" && !ev ? "sin eventos exactos" : "";
    } else if (!tr.n) {
      navInfo.textContent = "el θ no tiene eventos en el día";
    } else if (nav.k < 0 || !w) {
      navInfo.textContent = plural(tr.n, "evento", "eventos");
    } else {
      navInfo.textContent = "evento " + (nav.k + 1) + " / " + tr.n;
    }
    metrics.nav = { k: nav.k, window: w ? [w.start, w.end] : null };
  }

  // La navegación desplaza la ventana y conserva la escala que fijó el humano: el mismo ancho,
  // centrado en la ventana del evento y sin salirse del día.
  function goTo(k) {
    var tr = activeBlock();
    if (!tr || k < 0 || k >= tr.n) return;
    nav.k = k;
    var w = windowOf(k);
    var x = pricePlot.scales.x;
    var span = Math.min(DAY_S, x.max - x.min);
    var min = (w.start + w.end) / 2 - span / 2;
    min = Math.min(Math.max(0, min), DAY_S - span);
    renderNav(w);
    setX(min, min + span);
    pricePlot.redraw(false);
  }

  // Sin evento elegido, "siguiente" va al primero que arranca después del centro de la vista y
  // "anterior" al último que arrancó antes.
  function stepEvent(delta) {
    var tr = activeBlock();
    if (!tr || !tr.n) return;
    if (nav.k >= 0) return goTo(nav.k + delta);
    var x = pricePlot.scales.x;
    var centerMs = ((x.min + x.max) / 2) * 1000;
    var after = lowerBound(ev.ref, tr, Math.floor(centerMs) + 1); // primer evento con referencia > centro
    goTo(delta > 0 ? Math.min(after, tr.n - 1) : Math.max(after - 1, 0));
  }

  // Acción separada: pone la escala en la ventana del evento, con un margen a cada lado.
  function fitWindow() {
    if (nav.k < 0 || !nav.window) return;
    var w = nav.window;
    var margin = Math.max((w.end - w.start) * FIT_MARGIN, 0.001);
    setX(Math.max(0, w.start - margin), Math.min(DAY_S, w.end + margin));
  }

  prevBtn.addEventListener("click", function () { stepEvent(-1); });
  nextBtn.addEventListener("click", function () { stepEvent(1); });
  fitBtn.addEventListener("click", fitWindow);

  if (select.options.length) {
    select.selectedIndex = 0;
    selectTheta(select.value);
  } else {
    setReason("theta", "sin θ en el día");
    renderStatus();
  }

  /* ---------- Los tres paneles ---------- */

  var panels = $("panels");
  var AXIS_W = 76;
  var pricePlot = null;
  var confP = null;
  var volP = null;
  var plots = [];
  var hovered = null; // panel bajo el puntero: el único que muestra el tooltip
  var syncing = false;

  // Redibujo de la vista: del cambio (zoom, θ o tamaño) al último panel que ese cambio repinta.
  // Cada panel avisa al terminar; cuando están todos se publica el total (marco incluido).
  var redraw = { t0: null, pending: [] };

  function redrawStart(keys) {
    redraw.t0 = performance.now();
    redraw.pending = keys.slice();
  }

  function redrawPanelDone(u) {
    if (redraw.t0 === null || stale(u)) return;
    metrics.draw_ms = performance.now() - redraw.t0;
    var i = redraw.pending.indexOf(yKey(u));
    if (i >= 0) redraw.pending.splice(i, 1);
    if (redraw.pending.length) return;
    redraw.t0 = null;
    console.info("viz: redibujo en " + metrics.draw_ms.toFixed(1) + " ms (" + metrics.frame.ticks + " ticks en la vista)");
  }

  function onScale(u, key) {
    if (key !== "x" || plots.length < 3) return;
    var s = u.scales.x;
    if (!syncing) {
      redrawStart(["p", "c", "v"]);
      syncing = true;
      plots.forEach(function (other) {
        var t = other.scales.x;
        if (other !== u && (t.min !== s.min || t.max !== s.max)) other.setScale("x", { min: s.min, max: s.max });
      });
      syncing = false;
    }
    // Fuera del commit en curso: un setScale dentro de su hook no se aplica.
    queueMicrotask(function () { applyY(u); });
  }

  // El eje Y de cada panel sale del marco, o sea de lo que hay en la vista. uPlot calcula sus rangos
  // con la x anterior, así que aquí se fijan a mano, ya con la x nueva: la primera pasada de dibujo
  // de un cambio se salta (`stale`) y la que sigue pinta con el rango bueno.
  var Y = { p: yPrice, c: yConfirms, v: yVolume };

  function yKey(u) {
    return u === pricePlot ? "p" : u === confP ? "c" : "v";
  }

  function stale(u) {
    var key = yKey(u);
    var r = Y[key](u);
    var s = u.scales[key];
    return s.min !== r[0] || s.max !== r[1];
  }

  function applyY(u) {
    var key = yKey(u);
    var r = Y[key](u);
    var s = u.scales[key];
    if (s.min !== r[0] || s.max !== r[1]) u.setScale(key, { min: r[0], max: r[1] });
  }

  function resetZoom() {
    pricePlot.setScale("x", { min: 0, max: DAY_S });
  }

  function sizes() {
    var w = panels.clientWidth || 1200;
    var h = panels.clientHeight || 700;
    var hv = Math.round(h * SPLIT.volume);
    var hc = Math.round(h * SPLIT.confirms);
    return { w: w, hp: h - hv - hc, hc: hc, hv: hv };
  }

  // Los ejes Y salen del marco: lo que hay en la vista, no de una serie de uPlot.
  function yPrice(u) {
    var f = getFrame(u);
    var lo = f.pMin / scale;
    var hi = f.pMax / scale;
    var pad = (hi - lo) * 0.05 || 1;
    return [lo - pad, hi + pad];
  }

  function yVolume(u) {
    var f = getFrame(u);
    return [0, f.volMax > 0 ? f.volMax * 1.08 : 1];
  }

  // Enteros pequeños: el eje parte de 0 y llega al máximo con un poco de aire.
  function yConfirms(u) {
    var f = getFrame(u);
    return [0, f.ccMax > 0 ? Math.ceil(f.ccMax * 1.15) : 1];
  }

  function axisBase() {
    return {
      stroke: C.muted,
      font: "12px " + MONO,
      grid: { stroke: C.grid, width: 1 },
      ticks: { stroke: C.grid, width: 1 },
    };
  }

  // Desde 1 ms: la resolución de los datos. A un ms de ventana el eje marca cada ms.
  var X_INCRS = [
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
    1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 14400, 21600, 43200,
  ];

  function xAxis(labels) {
    var a = axisBase();
    a.scale = "x";
    a.size = labels ? 28 : 6;
    a.space = 90;
    a.incrs = X_INCRS;
    a.values = function (u, splits, axisIdx, space, incr) {
      return splits.map(function (v) {
        if (!labels) return "";
        if (incr >= 60) return clock(v, 0).slice(0, 5);
        if (incr >= 1) return clock(v, 0);
        return clock(v, incr >= 0.1 ? 1 : incr >= 0.01 ? 2 : 3);
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

  // uPlot pone los ejes, la selección y el cursor; las marcas las dibujan los hooks. Cada panel
  // lleva una serie sin trazo y dos puntos: lo mínimo para que sus escalas existan.
  function stub() {
    return [[0, DAY_S], [0, 1]];
  }

  function series(scaleKey) {
    return [{}, { scale: scaleKey, show: true, paths: function () { return null; }, points: { show: false } }];
  }

  function build() {
    var s = sizes();
    var shared = { legend: { show: false }, cursor: cursorOpts() };

    // El rango inicial es el del día completo, con el ancho que se estima antes de que exista el panel.
    var whole = { scales: { x: { min: 0, max: DAY_S } } };
    var yp = yPrice(whole);
    var yc = yConfirms(whole);
    var yv = yVolume(whole);

    pricePlot = new uPlot(
      Object.assign({}, shared, {
        width: s.w,
        height: s.hp,
        padding: [8, 12, 0, 0],
        scales: { x: { time: false, auto: false, min: 0, max: DAY_S }, p: { auto: false, min: yp[0], max: yp[1] } },
        axes: [xAxis(false), yAxis("p", fixed)],
        series: series("p"),
        hooks: {
          drawClear: [drawRegions],
          draw: [
            function (u) {
              if (stale(u)) return; // el eje Y se está poniendo al día: la pasada que sigue pinta
              drawPrice(u);
              drawMessages(u);
              var now = performance.now();
              if (metrics.first_paint_ms === null) {
                metrics.first_paint_ms = now;
                console.info(
                  "viz: primer trazo a " + now.toFixed(1) + " ms desde el inicio de la navegación (" +
                    (now - started).toFixed(1) + " ms desde que arrancó el script)"
                );
              }
              redrawPanelDone(u);
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
      stub(),
      $("price")
    );

    // Confirmaciones: la barra completa son los θ que confirman en el píxel y la marca intensa el
    // máximo de θ que confirman en el mismo instante. Se lee por longitud.
    confP = new uPlot(
      Object.assign({}, shared, {
        width: s.w,
        height: s.hc,
        padding: [4, 12, 0, 0],
        scales: { x: { time: false, auto: false, min: 0, max: DAY_S }, c: { auto: false, min: yc[0], max: yc[1] } },
        axes: [
          xAxis(false),
          Object.assign(
            yAxis("c", function (v) {
              return Number.isInteger(v) ? String(v) : "";
            }),
            { incrs: [1, 2, 5, 10, 20, 50, 100] }
          ),
        ],
        series: series("c"),
        hooks: { draw: [drawConfirms, redrawPanelDone], setCursor: [onCursor], setScale: [onScale] },
      }),
      stub(),
      $("confirms")
    );

    volP = new uPlot(
      Object.assign({}, shared, {
        width: s.w,
        height: s.hv,
        padding: [4, 12, 0, 0],
        scales: { x: { time: false, auto: false, min: 0, max: DAY_S }, v: { auto: false, min: yv[0], max: yv[1] } },
        axes: [
          xAxis(true),
          yAxis("v", function (v) {
            return v >= 10 ? v.toFixed(0) : v.toFixed(2);
          }),
        ],
        series: series("v"),
        hooks: { draw: [drawVolume, redrawPanelDone], setCursor: [onCursor], setScale: [onScale] },
      }),
      stub(),
      $("volume")
    );

    plots = [pricePlot, confP, volP];
    plots.forEach(function (u) {
      u.over.addEventListener("dblclick", resetZoom);
      u.over.addEventListener("mouseenter", function () { hovered = u; });
      u.over.addEventListener("mouseleave", function () { hovered = null; tip.hidden = true; });
    });
    renderNav(null);
  }

  function renderFooter(f) {
    var perPx = 1 / f.k;
    var text =
      plural(f.n, "tick", "ticks") + " en la vista · " +
      (perPx >= 1 ? perPx.toFixed(perPx >= 100 ? 0 : 1) + " ms" : (perPx * 1000).toFixed(0) + " µs") + " por píxel";
    $("f-view").textContent = text;
  }

  /* ---------- Arranque ---------- */

  var thetaT0 = null;
  loadTicks();
  loadEvents();
  renderStatus();
  if (!ticks) {
    panels.textContent = "Sin ticks: el día no se puede dibujar.";
    return;
  }
  build();

  select.addEventListener("change", function () {
    thetaT0 = performance.now();
    redrawStart(["p"]);
    selectTheta(select.value);
    pricePlot.redraw(false); // solo repinta: los eventos de los θ ya están en memoria
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") resetZoom();
  });

  window.addEventListener("resize", function () {
    var s = sizes();
    redrawStart(["p", "c", "v"]);
    pricePlot.setSize({ width: s.w, height: s.hp });
    confP.setSize({ width: s.w, height: s.hc });
    volP.setSize({ width: s.w, height: s.hv });
    // Los píxeles cambiaron: el marco y los ejes Y se calculan de nuevo con la misma ventana.
    frame = null;
    plots.forEach(applyY);
  });
})();
