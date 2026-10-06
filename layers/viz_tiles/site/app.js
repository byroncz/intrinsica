/* Vista de un día (TRD-viz §6.6, §6.7, §6.11 a §6.13, §7). Sin red: los tiles llegan en window.VIZ_DATA,
 * en base64 bajo su nombre. El navegador solo decodifica y hace las dos conversiones
 * de §7.3 (t / 1000 y p / price_scale, o null en el centinela); los tiempos de los eventos
 * (ms) los pasa a segundos para el eje X. Lo demás sale de los tiles tal cual: los conteos
 * y las confirmaciones multiescala llegan precalculados, el navegador no cuenta nada de
 * eso. Solo agrupa los eventos exactos de un θ por píxel (la marca de densidad).
 * Sin telemetría: las métricas van a la consola. */
(function () {
  "use strict";

  var DAY_S = 86400;
  var EMPTY = -2147483648; // centinela de p en una columna sin ticks (§7.3)
  var KNOWN_MAJOR = 1;
  // Desde este ancho (px CSS por columna) se dibujan, sobre el segmento mín–máx de una columna
  // con 3 ticks o más, sus puntos M4 (primero, mínimo, máximo y último: ticks reales) en su
  // instante exacto.
  var DOT_MIN_COL_PX = 5;
  var DOT_PX = 3; // lado de un punto, en px CSS
  var MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace';
  var C = {
    text: "#d7dee6",
    muted: "#9aa7b4",
    grid: "rgba(154, 167, 180, 0.16)",
    price: "#e6edf3",
    volume: "rgba(110, 138, 168, 0.85)",
    confirms: "rgba(86, 182, 194, 0.42)",
    simul: "rgb(86, 182, 194)",
    amber: "#e3b341",
    up: "63, 185, 80",
    down: "248, 81, 73",
  };
  // Estados de dirección (§7.4): solo para el tooltip. Las franjas salen de los eventos exactos.
  var STATES = {
    0: { text: "sin evento" },
    1: { text: "confirmación alza" },
    2: { text: "overshoot alza" },
    3: { text: "confirmación baja" },
    4: { text: "overshoot baja" },
  };
  // Banderas de un evento (§7.5).
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
  // Marca del hueco (columnas sin ticks): línea punteada de 1 px a media altura. Ninguna franja
  // DC es punteada, así de fina ni va al centro: un hueco nunca se lee como una confirmación.
  var GAP_MARK = { px: 1, dash: [4, 3] };
  var TIP_MAX_LINES = 12;

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
  if (!(major <= KNOWN_MAJOR)) {
    setReason("version", "tiles_version " + data.tiles_version + " no soportada: se rechaza el día");
    renderStatus();
    $("panels").textContent = "Versión de tiles no soportada (" + data.tiles_version + ").";
    return;
  }
  if (missingThetas.length) {
    setReason("missing", missingThetas.length + " θ sin datos en L2");
  }

  // Nombre del tile de un tipo y nivel; undefined si el índice no lo lista (tiles anteriores a 1.2.0).
  function nameOf(kind, w) {
    var byLevel = index[kind];
    return byLevel ? byLevel[w] : undefined;
  }

  // Un tile que el índice lista y la página no trae es un hueco visible, no un nivel que se omite en silencio.
  levels.forEach(function (w) {
    ["price", "volume", "dir", "count", "confirms", "simul"].forEach(function (kind) {
      if (kind === "dir" && !thetas.length) return;
      var name = nameOf(kind, w);
      if (name === undefined) setReason("tile:" + kind + "-" + w, "el índice no lista " + kind + " del nivel " + w);
      else if (files[name] === undefined) setReason("tile:" + name, "falta " + name);
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

  function decodeCount(name, w) {
    var bytes = bytesOf(name);
    if (!bytes) return null;
    if (bytes.length !== 4 * w) return bad(name, "tamaño inesperado");
    return new Uint32Array(bytes.buffer, 0, w);
  }

  function decodeBytes(name, w, perColumn) {
    var bytes = bytesOf(name);
    if (!bytes) return null;
    if (bytes.length !== perColumn * w) return bad(name, "tamaño inesperado");
    return bytes;
  }

  // Eventos exactos de todos los θ del día (§7.5): cuatro secciones de N valores. El θ k ocupa
  // de events_offset a events_offset + events - 1 en cada una.
  var ev = null;

  function loadEvents() {
    var name = index.events;
    if (!name) {
      setReason("events", "el índice no lista los eventos exactos (tiles_version " + data.tiles_version + ")");
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
  }

  var cache = {}; // w -> nivel decodificado

  // cache[w] === false marca un nivel que no se pudo decodificar: se salta como uno ausente.
  function hasLevel(w) {
    if (cache[w] !== undefined) return cache[w] !== false;
    return files[nameOf("price", w)] !== undefined;
  }

  function loadLevel(w) {
    if (cache[w] !== undefined) return cache[w] || null;
    var t0 = performance.now();
    var price = decodePrice(nameOf("price", w), w);
    if (!price) {
      cache[w] = false;
      return null;
    }
    var vol = decodeVolume(nameOf("volume", w), w);
    var dir = thetas.length ? decodeBytes(nameOf("dir", w), w, thetas.length) : null;
    var cnt = decodeCount(nameOf("count", w), w);
    var confirms = decodeBytes(nameOf("confirms", w), w, 1);
    var simul = decodeBytes(nameOf("simul", w), w, 1);
    var colStart = new Float64Array(w);
    for (var c = 0; c < w; c++) colStart[c] = (c * DAY_S) / w; // posición de la columna
    // Sin un tile no hay barras: nulls, nunca ceros (principio 6).
    var nulls = function () { return new Array(w).fill(null); };
    var level = {
      w: w,
      colDur: DAY_S / w,
      price: price,
      vol: vol,
      dir: dir,
      cnt: cnt,
      confirms: confirms,
      simul: simul,
      priceData: [price.x, price.y],
      volData: [colStart, vol || nulls()],
      confData: [colStart, confirms || nulls(), simul || nulls()],
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

  var cur = null; // nivel en pantalla
  var price = null;
  var conf = null;
  var vol = null;
  var plots = [];
  var hovered = null; // panel bajo el puntero: el único que muestra el tooltip

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

  function fixed(v) {
    return v.toFixed(decimals);
  }

  function plural(n, one, many) {
    return n + " " + (n === 1 ? one : many);
  }

  /* ---------- Eventos exactos del θ activo ---------- */

  // El bloque del θ activo en `ev`: {off, n}, o null si no hay θ con bloque o faltan los eventos.
  function block(k) {
    if (!ev || k < 0 || !thetas[k]) return null;
    return { off: thetas[k].events_offset, n: thetas[k].events };
  }

  function activeBlock() {
    return sel.kind === "t" ? block(sel.k) : null;
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

  function emptyColumn(level, c) {
    return level.price.y[4 * c] === null;
  }

  /* ---------- Dibujo de las regiones (en el lienzo de uPlot, bajo la serie) ---------- */

  function vline(ctx, x, b, dpr, style) {
    ctx.fillStyle = style;
    ctx.fillRect(Math.round(x - dpr / 2), b.top, dpr, b.height);
  }

  // Huecos: columnas sin ticks. Marcador y texto; nunca se rellena.
  function drawGaps(u, ctx, b, dpr) {
    var level = cur;
    var xs = u.scales.x;
    var c0 = Math.max(0, Math.floor(xs.min / level.colDur));
    var c1 = Math.min(level.w - 1, Math.ceil(xs.max / level.colDur) - 1);
    var c = c0;
    while (c <= c1) {
      if (!emptyColumn(level, c)) {
        c++;
        continue;
      }
      var end = c;
      while (end + 1 <= c1 && emptyColumn(level, end + 1)) end++;
      var x0 = u.valToPos(c * level.colDur, "x", true);
      var x1 = u.valToPos((end + 1) * level.colDur, "x", true);
      var wpx = Math.max(1, x1 - x0);
      ctx.fillStyle = "rgba(227, 179, 65, 0.10)";
      ctx.fillRect(x0, b.top, wpx, b.height);
      ctx.strokeStyle = C.amber;
      ctx.lineWidth = GAP_MARK.px * dpr;
      ctx.setLineDash([GAP_MARK.dash[0] * dpr, GAP_MARK.dash[1] * dpr]);
      ctx.beginPath();
      ctx.moveTo(x0, b.top + b.height / 2);
      ctx.lineTo(x0 + wpx, b.top + b.height / 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.amber;
      if (wpx > 70 * dpr) {
        ctx.textAlign = "center";
        ctx.fillText("◇ sin ticks", x0 + wpx / 2, b.top + b.height - 8 * dpr);
      } else if (wpx > 14 * dpr) {
        ctx.textAlign = "center";
        ctx.fillText("◇", x0 + wpx / 2, b.top + b.height - 8 * dpr);
      }
      c = end + 1;
    }
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
      // Confirmación: línea del color del evento. Extremo: frontera de 1 px compartida con el
      // evento que sigue (el tick extremo cierra su evento; el siguiente arranca en el tick que
      // lo sigue). Ni una ni otra se dibujan si el tiempo quedó recortado al borde del día; el
      // extremo provisional lo marca la línea ámbar de abajo.
      if (!(f & F_CONFIRM_CLIPPED)) vline(ctx, xc, b, dpr, "rgb(" + (up ? C.up : C.down) + ")");
      if (!(f & (F_EXTREME_CLIPPED | F_PROVISIONAL))) vline(ctx, x1, b, dpr, C.text);
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
      vline(ctx, left, b, dpr, C.text);
      vline(ctx, left + width, b, dpr, C.text);
      ctx.fillStyle = C.text;
      ctx.fillText(grp.n + " eventos", left + width + 4 * dpr, b.top + (22 + 14 * (row % 3)) * dpr);
      row++;
      out.groups++;
      out.grouped += grp.n;
      if (grp.n > out.max) out.max = grp.n;
    });

    // El evento elegido con la navegación: un marco sobre su intervalo.
    if (nav.k >= 0 && nav.k < tr.n) {
      var s = tr.off + nav.k;
      var sx0 = u.valToPos(ev.ref[s] / 1000, "x", true);
      var sx1 = u.valToPos(ev.ext[s] / 1000, "x", true);
      ctx.strokeStyle = C.text;
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

    drawGaps(u, ctx, b, dpr);
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

  /* ---------- Serie de precio: puntos o segmentos, nunca velas ---------- */

  // Lo que se dibuja es un tick o la envolvente exacta de ticks (§6.4):
  //  - columna con 1 o 2 ticks: M4 son todos sus ticks, un punto por tick;
  //  - columna con 3 ticks o más: siempre el segmento del mínimo al máximo, la unión de los
  //    píxeles que ocuparían sus puntos; si mínimo y máximo son iguales, el tramo horizontal del
  //    primer al último tick;
  //  - y, si además es ancha (>= DOT_MIN_COL_PX), sus cuatro puntos M4 (ticks reales) en su
  //    instante exacto, sobre el segmento: nunca solo cuatro puntos aislados, que se leerían
  //    como "4 ticks" en una columna de miles.
  // Nada une una columna con la vecina: nadie midió lo que hay entre ellas (principio 6).
  var drawMode = "segments"; // "segments" | "points"

  // Ancho en pantalla de una columna del nivel actual, en px CSS.
  function columnPx(u) {
    var xs = u.scales.x;
    return (u.bbox.width / (window.devicePixelRatio || 1)) * (cur.colDur / (xs.max - xs.min));
  }

  function setDrawMode(mode, drawn) {
    metrics.price_draw = { mode: mode, segments: drawn.segments, points: drawn.points };
    if (mode === drawMode) return;
    drawMode = mode;
    console.info("viz: precio en " + (mode === "points" ? "segmentos mín–máx y puntos M4 por columna" : "segmentos mín–máx por columna"));
    renderFooter();
  }

  function pricePaths(u, seriesIdx, idx0, idx1) {
    var level = cur;
    var dpr = window.devicePixelRatio || 1;
    var xs = u.scales.x;
    var y = level.price.y;
    var t = level.price.x;
    var wide = columnPx(u) >= DOT_MIN_COL_PX;
    var c0 = Math.max(0, Math.floor(xs.min / level.colDur));
    var c1 = Math.min(level.w - 1, Math.ceil(xs.max / level.colDur) - 1);
    var half = (DOT_PX * dpr) / 2;
    var path = new Path2D();
    var drawn = { segments: 0, points: 0 };

    function dot(tt, pp) {
      var px = u.valToPos(tt, "x", true);
      var py = u.valToPos(pp, "p", true);
      path.rect(px - half, py - half, 2 * half, 2 * half);
      drawn.points++;
    }

    // Los puntos M4 de la columna que empieza en el índice k, sin repetir los que coinciden.
    function m4Dots(k) {
      for (var j = 0; j < 4; j++) {
        var dup = false;
        for (var q = 0; q < j; q++) {
          if (t[k + q] === t[k + j] && y[k + q] === y[k + j]) dup = true;
        }
        if (!dup) dot(t[k + j], y[k + j]);
      }
    }

    for (var c = c0; c <= c1; c++) {
      if (emptyColumn(level, c)) continue; // hueco: lo marca drawRegions, aquí no se dibuja nada
      var k = 4 * c;
      var n = level.cnt ? level.cnt[c] : 3; // sin el tile de conteo no se sabe si M4 son todos los ticks
      if (n <= 2) {
        m4Dots(k);
        continue;
      }
      var lo = Infinity;
      var hi = -Infinity;
      for (var m = k; m < k + 4; m++) {
        if (y[m] < lo) lo = y[m];
        if (y[m] > hi) hi = y[m];
      }
      if (lo === hi) {
        // Todos los ticks al mismo precio: del primero al último, a ese precio.
        var yy = u.valToPos(lo, "p", true);
        path.moveTo(u.valToPos(t[k], "x", true), yy);
        path.lineTo(Math.max(u.valToPos(t[k + 3], "x", true), u.valToPos(t[k], "x", true) + dpr), yy);
      } else {
        var xc = u.valToPos((c + 0.5) * level.colDur, "x", true);
        var yTop = u.valToPos(hi, "p", true);
        var yBottom = u.valToPos(lo, "p", true);
        if (yBottom - yTop < dpr) {
          yTop -= dpr / 2;
          yBottom += dpr / 2; // al menos un píxel: un rango no se vuelve invisible
        }
        path.moveTo(xc, yTop);
        path.lineTo(xc, yBottom);
        drawn.segments++;
      }
      if (wide) m4Dots(k);
    }
    setDrawMode(wide ? "points" : "segments", drawn);
    return { stroke: path, fill: null, clip: null, band: null, gaps: null, flags: 3 };
  }

  function drawMessages(u) {
    var msg = null;
    if (sel.kind === "m") msg = "⚠ θ " + sel.theta + ": sin datos en L2 (hueco, no se rellena)";
    else if (sel.kind === "t" && !ev) msg = "⚠ faltan los eventos exactos: no se dibujan franjas";
    else if (sel.kind === "t" && !cur.dir) msg = "⚠ falta el tile de dirección de este nivel";
    if (msg) drawNote(u, msg);
  }

  function drawVolumeMessage(u) {
    if (!cur.vol) drawNote(u, "⚠ falta el tile de volumen de este nivel");
  }

  function drawConfirmsMessage(u) {
    var missing = [];
    if (!cur.confirms) missing.push("confirms");
    if (!cur.simul) missing.push("simul");
    if (missing.length) drawNote(u, "⚠ falta el tile " + missing.join(" y ") + " de este nivel");
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

  // Las confirmaciones de todos los θ dentro de la cubeta [a, b) segundos, a partir de los
  // eventos exactos: [{theta, ms}] en orden de hora.
  function confirmsIn(a, b) {
    var out = [];
    if (!ev) return out;
    var lo = Math.ceil(a * 1000);
    var hi = Math.ceil(b * 1000);
    thetas.forEach(function (t, k) {
      var tr = block(k);
      for (var i = lowerBound(ev.conf, tr, lo); i < tr.n; i++) {
        var e = tr.off + i;
        if (ev.conf[e] >= hi) break;
        if (!(ev.flags[e] & F_CONFIRM_CLIPPED)) out.push({ theta: t.theta, ms: ev.conf[e] });
      }
    });
    out.sort(function (p, q) { return p.ms - q.ms || (p.theta < q.theta ? -1 : 1); });
    return out;
  }

  function bucketLines(level, c, dec) {
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
    lines.push(level.cnt ? plural(level.cnt[c], "tick", "ticks") : "ticks: ⚠ falta el tile");
    return lines;
  }

  function thetaLines(level, c) {
    var lines = [];
    if (sel.kind === "t" && !level.dir) {
      lines.push("θ " + sel.theta + ": ⚠ falta el tile de dirección");
    } else if (sel.kind === "t") {
      var code = level.dir[sel.k * level.w + c];
      lines.push("θ " + sel.theta + ": " + (STATES[code] ? STATES[code].text : "reservado (" + code + ")"));
    } else if (sel.kind === "m") {
      lines.push("θ " + sel.theta + ": sin datos");
    }
    var tr = activeBlock();
    if (tr) {
      // Los eventos enteros dentro de la cubeta (de su referencia a su extremo): con varios, una
      // franja por columna no los distingue.
      var a = Math.ceil(c * level.colDur * 1000);
      var z = Math.ceil((c + 1) * level.colDur * 1000);
      var first = lowerBound(ev.ref, tr, a);
      var n = Math.min(lowerBound(ev.ref, tr, z), lowerBound(ev.ext, tr, z)) - first;
      if (n > 0) lines.push("eventos del θ dentro de la cubeta: " + n);
    }
    return lines;
  }

  function confirmLines(level, c, dec) {
    var lines = [clock(c * level.colDur, dec) + " – " + clock((c + 1) * level.colDur, dec) + " UTC"];
    lines.push(
      level.confirms ? "θ que confirman: " + level.confirms[c] : "θ que confirman: ⚠ falta el tile"
    );
    lines.push(
      level.simul ? "máx. en el mismo instante: " + level.simul[c] : "máx. en el mismo instante: ⚠ falta el tile"
    );
    var list = confirmsIn(c * level.colDur, (c + 1) * level.colDur);
    list.slice(0, TIP_MAX_LINES).forEach(function (it) {
      lines.push("θ " + it.theta + "  " + clock(it.ms / 1000, 3));
    });
    if (list.length > TIP_MAX_LINES) lines.push("… y " + (list.length - TIP_MAX_LINES) + " más");
    return lines;
  }

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
    var lines =
      u === conf ? confirmLines(level, c, dec) : bucketLines(level, c, dec).concat(thetaLines(level, c));
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
    price.setScale("x", { min: min, max: max });
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
      navInfo.textContent = plural(tr.n, "evento", "eventos") + " en el día";
    } else {
      var e = tr.off + nav.k;
      var up = (ev.flags[e] & F_UP) !== 0;
      navInfo.textContent =
        "evento " + (nav.k + 1) + " de " + tr.n + " · " + (up ? "alza" : "baja") + " · referencia " +
        clock(ev.ref[e] / 1000, 3) + " · confirmación " + clock(ev.conf[e] / 1000, 3) + " · extremo " +
        clock(ev.ext[e] / 1000, 3) + " · ventana " + clock(w.start, 3) + " a " + clock(w.end, 3) +
        (w.notes.length ? " · " + w.notes.join(" · ") : "");
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
    var x = price.scales.x;
    var span = Math.min(DAY_S, x.max - x.min);
    var min = (w.start + w.end) / 2 - span / 2;
    min = Math.min(Math.max(0, min), DAY_S - span);
    renderNav(w);
    setX(min, min + span);
    price.redraw(false);
  }

  // Sin evento elegido, "siguiente" va al primero que arranca después del centro de la vista y
  // "anterior" al último que arrancó antes.
  function stepEvent(delta) {
    var tr = activeBlock();
    if (!tr || !tr.n) return;
    if (nav.k >= 0) return goTo(nav.k + delta);
    var x = price.scales.x;
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

  var syncing = false;
  var levelQueued = false;

  function onScale(u, key) {
    if (key !== "x" || plots.length < 3) return;
    var s = u.scales.x;
    if (!syncing) {
      syncing = true;
      plots.forEach(function (other) {
        var t = other.scales.x;
        if (other !== u && (t.min !== s.min || t.max !== s.max)) other.setScale("x", { min: s.min, max: s.max });
      });
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
    var level = null;
    while (w !== null && w !== cur.w && level === null) {
      level = loadLevel(w);
      if (level === null) w = pickLevel(span); // el nivel quedó inutilizable: el siguiente
    }
    if (level === null) return;
    // Al hacer zoom: el nivel más fino que cubre el rango visible, ya en memoria.
    cur = level;
    price.setData(level.priceData, false);
    conf.setData(level.confData, false);
    vol.setData(level.volData, false);
    plots.forEach(function (u) { u.setScale("x", { min: x.min, max: x.max }); });
    renderFooter();
  }

  function resetZoom() {
    price.setScale("x", { min: 0, max: DAY_S });
  }

  function sizes() {
    var w = panels.clientWidth || 1200;
    var h = panels.clientHeight || 700;
    var hv = Math.round(h * SPLIT.volume);
    var hc = Math.round(h * SPLIT.confirms);
    return { w: w, hp: h - hv - hc, hc: hc, hv: hv };
  }

  function yPrice(u, min, max) {
    if (!isFinite(min) || !isFinite(max)) return [0, 1];
    var pad = (max - min) * 0.05 || 1;
    return [min - pad, max + pad];
  }

  function yVolume(u, min, max) {
    return [0, isFinite(max) && max > 0 ? max * 1.08 : 1];
  }

  // Enteros pequeños: el eje parte de 0 y llega al máximo con un poco de aire.
  function yConfirms(u, min, max) {
    return [0, isFinite(max) && max > 0 ? Math.ceil(max * 1.15) : 1];
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
        series: [{}, { scale: "p", stroke: C.price, width: 1.5, points: { show: false }, paths: pricePaths }],
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

    // Confirmaciones multiescala: la barra completa son los θ que confirman en la columna y la
    // marca intensa el máximo de θ que confirman en el mismo instante. Se lee por longitud.
    var bars = uPlot.paths.bars({ size: [0.9, Infinity, 1], align: 1 });
    conf = new uPlot(
      Object.assign({}, shared, {
        width: s.w,
        height: s.hc,
        padding: [4, 12, 0, 0],
        scales: { x: { time: false, auto: false, min: 0, max: DAY_S }, c: { range: yConfirms } },
        axes: [
          xAxis(false),
          Object.assign(
            yAxis("c", function (v) {
              return Number.isInteger(v) ? String(v) : "";
            }),
            { incrs: [1, 2, 5, 10, 20, 50, 100] }
          ),
        ],
        series: [
          {},
          { scale: "c", stroke: C.confirms, fill: C.confirms, width: 0, points: { show: false }, paths: bars },
          { scale: "c", stroke: C.simul, fill: C.simul, width: 0, points: { show: false }, paths: bars },
        ],
        hooks: { draw: [drawConfirmsMessage], setCursor: [onCursor], setScale: [onScale] },
      }),
      cur.confData,
      $("confirms")
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
            paths: bars,
          },
        ],
        hooks: { draw: [drawVolumeMessage], setCursor: [onCursor], setScale: [onScale] },
      }),
      cur.volData,
      $("volume")
    );

    plots = [price, conf, vol];
    plots.forEach(function (u) {
      u.over.addEventListener("dblclick", resetZoom);
      u.over.addEventListener("mouseenter", function () { hovered = u; });
      u.over.addEventListener("mouseleave", function () { hovered = null; tip.hidden = true; });
    });
    renderNav(null);
  }

  function renderFooter() {
    var dur = cur.colDur;
    $("f-level").textContent =
      "nivel " + cur.w + " · " + (Number.isInteger(dur) ? dur : dur.toFixed(2)) + " s por columna" +
      " · puntos M4 sobre el segmento desde " + DOT_MIN_COL_PX + " px por columna · precio: " +
      (drawMode === "points" ? "segmentos y puntos" : "segmentos mín–máx");
  }

  /* ---------- Arranque ---------- */

  loadEvents();
  var thetaT0 = null;
  var first = pickLevel(DAY_S);
  cur = null;
  while (first !== null && cur === null) {
    cur = loadLevel(first);
    if (cur === null) first = pickLevel(DAY_S); // el nivel quedó inutilizable: el siguiente
  }
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
    conf.setSize({ width: s.w, height: s.hc });
    vol.setSize({ width: s.w, height: s.hv });
    queueLevel();
  });
})();
