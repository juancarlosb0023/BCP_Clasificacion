
(function () {
  "use strict";

  const DATA = {
    state: "./data/current_asset_state.json",
    alerts: "./data/dashboard_condition_alerts_active.csv",
    trendCsv: "./data/dashboard_condition_trend_summary.csv",
    contribCsv: "./data/dashboard_condition_contributions_top_by_window.csv",
  };

  async function fetchJson(path) {
    const res = await fetch(path, { cache: "no-store" });
    if (!res.ok) throw new Error("No se pudo cargar " + path + " (" + res.status + ")");
    return res.json();
  }

  async function fetchText(path) {
    const res = await fetch(path, { cache: "no-store" });
    if (!res.ok) throw new Error("No se pudo cargar " + path + " (" + res.status + ")");
    return res.text();
  }

  /** Parser CSV simple con soporte de comillas RFC4180-lite */
  function parseCsv(text) {
    const rows = [];
    let i = 0;
    const n = text.length;
    let row = [];
    let field = "";
    let inQuotes = false;
    while (i < n) {
      const c = text[i];
      if (inQuotes) {
        if (c === '"') {
          if (text[i + 1] === '"') {
            field += '"';
            i += 2;
            continue;
          }
          inQuotes = false;
          i++;
          continue;
        }
        field += c;
        i++;
        continue;
      }
      if (c === '"') {
        inQuotes = true;
        i++;
        continue;
      }
      if (c === ",") {
        row.push(field);
        field = "";
        i++;
        continue;
      }
      if (c === "\r") {
        i++;
        continue;
      }
      if (c === "\n") {
        row.push(field);
        rows.push(row);
        row = [];
        field = "";
        i++;
        continue;
      }
      field += c;
      i++;
    }
    if (field.length > 0 || row.length > 0) {
      row.push(field);
      rows.push(row);
    }
    if (rows.length === 0) return { headers: [], rows: [] };
    const headers = rows[0].map((h) => h.trim());
    const body = [];
    for (let r = 1; r < rows.length; r++) {
      if (rows[r].length === 1 && rows[r][0] === "") continue;
      body.push(rows[r]);
    }
    return { headers, rows: body };
  }

  function formatNumber(value, decimals) {
    decimals = decimals === undefined ? 2 : decimals;
    const x = Number(value);
    if (!isFinite(x)) return "N/D";
    return x.toFixed(decimals);
  }

  function formatPercent(value) {
    const x = Number(value);
    if (!isFinite(x)) return "N/D";
    let p = x;
    if (p >= 0 && p <= 1) p = p * 100;
    return p.toFixed(2) + "%";
  }

  function safeText(value) {
    if (value === null || value === undefined) return "N/D";
    const s = String(value).trim();
    return s.length ? s : "N/D";
  }

  function stateCssClass(conditionState) {
    const s = String(conditionState || "").toLowerCase();
    if (s === "normal") return "normal";
    if (s === "attention") return "attention";
    if (s === "high") return "high";
    return "unknown";
  }

  function showError(message) {
    const el = document.getElementById("error-banner");
    if (!el) return;
    el.textContent = message;
    el.classList.remove("hidden");
  }

  function appendError(message) {
    const el = document.getElementById("error-banner");
    if (!el) return;
    el.textContent = (el.textContent ? el.textContent + "\n" : "") + message;
    el.classList.remove("hidden");
  }

  function rowToObject(headers, row) {
    const o = {};
    for (let c = 0; c < headers.length; c++) {
      o[headers[c]] = row[c] !== undefined ? row[c] : "";
    }
    return o;
  }

  function renderKpiCards(state) {
    const grid = document.getElementById("kpi-grid");
    if (!grid) return;
    grid.innerHTML = "";
    const trend = state.trend || {};
    const alerts = Array.isArray(state.alerts_active) ? state.alerts_active : [];
    let nAtt = 0;
    let nHigh = 0;
    for (let i = 0; i < alerts.length; i++) {
      const lv = String(alerts[i].alert_level || "").toLowerCase();
      if (lv === "attention") nAtt++;
      if (lv === "high") nHigh++;
    }
    const cards = [
      { label: "Estado actual", value: safeText(state.condition_state), key: "condition_state" },
      { label: "Condition index", value: formatNumber(state.condition_index), key: "ci" },
      { label: "Health index", value: formatNumber(state.health_index), key: "hi" },
      { label: "Batch real", value: safeText(state.batch_real), key: "br" },
      { label: "Batch predicho", value: safeText(state.batch_predicted), key: "bp" },
      { label: "Confianza clasificación", value: formatPercent(state.classification_confidence), key: "cf" },
      { label: "Tendencia", value: safeText(trend.trend_direction), key: "tr" },
      { label: "Alertas attention", value: String(nAtt), key: "aa" },
      { label: "Alertas high", value: String(nHigh), key: "ah" },
    ];
    const cls = stateCssClass(state.condition_state);
    for (let k = 0; k < cards.length; k++) {
      const card = document.createElement("div");
      card.className = "kpi-card " + cls;
      card.innerHTML =
        '<div class="label"></div><div class="value"></div>';
      card.querySelector(".label").textContent = cards[k].label;
      card.querySelector(".value").textContent = cards[k].value;
      grid.appendChild(card);
    }
  }

  function renderCurrentState(state) {
    const ts = document.getElementById("state-timestamp");
    if (ts) ts.textContent = "Estado (UTC): " + safeText(state.state_timestamp);

    const cw = state.current_window || {};
    const block = document.getElementById("current-window-block");
    if (!block) return;
    const rows = [
      ["window_id", safeText(cw.window_id)],
      ["window_start", safeText(cw.window_start)],
      ["window_end", safeText(cw.window_end)],
      ["baseline_used", safeText(state.baseline_used)],
      ["assessment_method", safeText(state.assessment_method)],
      ["classification_margin_top2", formatNumber(state.classification_margin_top2, 4)],
    ];
    block.innerHTML = "";
    for (let i = 0; i < rows.length; i++) {
      const div = document.createElement("div");
      div.className = "info-row";
      div.innerHTML = '<span class="k"></span><span class="v"></span>';
      div.querySelector(".k").textContent = rows[i][0];
      div.querySelector(".v").textContent = rows[i][1];
      block.appendChild(div);
    }
  }

  function renderSemaphore(state) {
    const el = document.getElementById("semaphore-block");
    if (!el) return;
    const ci = Number(state.condition_index);
    const cls = stateCssClass(state.condition_state);
    let band = "unknown";
    let bandLabel = "N/D";
    if (isFinite(ci)) {
      if (ci < 50) {
        band = "normal";
        bandLabel = "Normal (< 50)";
      } else if (ci < 80) {
        band = "attention";
        bandLabel = "Atención (50 – < 80)";
      } else {
        band = "high";
        bandLabel = "Alto (≥ 80)";
      }
    }
    const pct = isFinite(ci) ? Math.min(100, Math.max(0, ci)) : 0;
    el.innerHTML =
      '<div class="sem-bar"><div class="sem-marker" id="sem-marker"></div></div>' +
      '<div class="sem-labels"><span>0</span><span>50</span><span>80</span><span>100</span></div>' +
      '<div class="sem-value">Índice: ' +
      formatNumber(state.condition_index) +
      " — Banda: " +
      bandLabel +
      "</div>" +
      '<p class="note-inline">Clase de estado declarada: <strong>' +
      safeText(state.condition_state) +
      "</strong> (" +
      cls +
      ")</p>";
    const mk = el.querySelector("#sem-marker");
    if (mk) mk.style.left = pct + "%";
  }

  function renderAlerts(rowsParsed) {
    const container = document.getElementById("alerts-container");
    if (!container) return;
    const headers = rowsParsed.headers;
    const rows = rowsParsed.rows;
    if (!headers.length) {
      container.innerHTML = '<p class="empty-msg">No hay datos de alertas.</p>';
      return;
    }
    if (!rows.length) {
      container.innerHTML =
        '<p class="empty-msg">No hay alertas activas en la última ventana exportada.</p>';
      return;
    }
    const want = [
      "alert_id",
      "raw_variable",
      "component",
      "family",
      "condition_score",
      "weighted_score",
      "alert_level",
      "persistence_count",
      "assessment_method",
    ];
    const cols = want.filter(function (c) {
      return headers.indexOf(c) >= 0;
    });
    if (!cols.length) {
      cols.push.apply(cols, headers);
    }
    let html = '<table class="data-table"><thead><tr>';
    for (let c = 0; c < cols.length; c++) {
      html += "<th>" + cols[c] + "</th>";
    }
    html += "</tr></thead><tbody>";
    for (let r = 0; r < rows.length; r++) {
      const obj = rowToObject(headers, rows[r]);
      html += "<tr>";
      for (let c = 0; c < cols.length; c++) {
        const key = cols[c];
        let val = obj[key];
        if (
          key === "condition_score" ||
          key === "weighted_score"
        ) {
          val = formatNumber(val);
        } else {
          val = safeText(val);
        }
        html += "<td>" + val + "</td>";
      }
      html += "</tr>";
    }
    html += "</tbody></table>";
    container.innerHTML = html;
  }

  function renderTopContributors(state, contribParsed) {
    const container = document.getElementById("contributors-container");
    if (!container) return;
    let list = Array.isArray(state.top_condition_drivers) ? state.top_condition_drivers : [];
    let fromJson = list.length > 0;
    if (!fromJson) {
      const cw = state.current_window || {};
      const wid = cw.window_id;
      const headers = contribParsed.headers;
      const rows = contribParsed.rows;
      if (headers.length && rows.length && headers.indexOf("window_id") >= 0) {
        list = [];
        for (let r = 0; r < rows.length; r++) {
          const o = rowToObject(headers, rows[r]);
          if (String(o.window_id) === String(wid)) list.push(o);
        }
        list.sort(function (a, b) {
          return Number(a.rank) - Number(b.rank);
        });
      }
    }
    if (!list.length) {
      container.innerHTML = '<p class="empty-msg">Sin datos de contribuciones.</p>';
      return;
    }
    const cols = [
      "rank",
      "raw_variable",
      "component",
      "family",
      "condition_score",
      "weighted_score",
      "share_of_condition_index_percent",
    ];
    let html = '<table class="data-table"><thead><tr>';
    for (let c = 0; c < cols.length; c++) {
      html += "<th>" + cols[c] + "</th>";
    }
    html += "</tr></thead><tbody>";
    for (let i = 0; i < list.length; i++) {
      const o = list[i];
      html += "<tr>";
      for (let c = 0; c < cols.length; c++) {
        const key = cols[c];
        let val = o[key];
        if (key === "condition_score" || key === "weighted_score" || key === "share_of_condition_index_percent") {
          val = formatNumber(val);
        } else {
          val = safeText(val);
        }
        html += "<td>" + val + "</td>";
      }
      html += "</tr>";
    }
    html += "</tbody></table>";
    if (!fromJson) {
      html +=
        '<p class="note-inline">Fuente: dashboard_condition_contributions_top_by_window.csv (JSON sin top_condition_drivers).</p>';
    }
    container.innerHTML = html;
  }

  function renderTrend(state, trendParsed) {
    const container = document.getElementById("trend-container");
    if (!container) return;
    let t = state.trend && typeof state.trend === "object" ? state.trend : null;
    if (!t || !Object.keys(t).length) {
      const h = trendParsed.headers;
      const r0 = trendParsed.rows[0];
      if (h.length && r0) t = rowToObject(h, r0);
    }
    if (!t) {
      container.innerHTML = '<p class="empty-msg">Sin datos de tendencia.</p>';
      return;
    }
    const keys = [
      "n_windows_used",
      "first_window_id",
      "last_window_id",
      "condition_index_first",
      "condition_index_last",
      "condition_index_delta",
      "slope_per_window",
      "slope_per_hour",
      "rolling_mean_last",
      "rolling_mean_previous",
      "trend_direction",
    ];
    let html = '<table class="data-table"><tbody>';
    for (let i = 0; i < keys.length; i++) {
      const k = keys[i];
      let v = t[k];
      if (
        k.indexOf("condition_index") === 0 ||
        k.indexOf("slope") === 0 ||
        k.indexOf("rolling_mean") === 0
      ) {
        v = formatNumber(v);
      } else {
        v = safeText(v);
      }
      html += "<tr><th>" + k + "</th><td>" + v + "</td></tr>";
    }
    html += "</tbody></table>";
    container.innerHTML = html;
  }

  function renderMethodology(state) {
    const el = document.getElementById("method-en");
    if (!el) return;
    const note =
      state.methodological_note ||
      "Exploratory condition assessment based on data-driven percentile thresholds; not a normative fault diagnosis.";
    el.textContent = note;
  }

  async function initDashboard() {
    let state;
    try {
      state = await fetchJson(DATA.state);
    } catch (e) {
      showError(String(e.message || e));
      return;
    }
    try {
      renderCurrentState(state);
      renderKpiCards(state);
      renderSemaphore(state);
      renderMethodology(state);
    } catch (e) {
      appendError("Error al renderizar estado: " + String(e.message || e));
    }

    let alertsParsed = { headers: [], rows: [] };
    try {
      const txt = await fetchText(DATA.alerts);
      alertsParsed = parseCsv(txt);
      renderAlerts(alertsParsed);
    } catch (e) {
      appendError("Alertas: " + String(e.message || e));
      const ac = document.getElementById("alerts-container");
      if (ac) ac.innerHTML = '<p class="empty-msg">No se pudieron cargar las alertas.</p>';
    }

    let contribParsed = { headers: [], rows: [] };
    try {
      const ctext = await fetchText(DATA.contribCsv);
      contribParsed = parseCsv(ctext);
    } catch (e) {
      appendError("Contribuciones CSV: " + String(e.message || e));
    }
    try {
      renderTopContributors(state, contribParsed);
    } catch (e) {
      appendError("Contribuciones: " + String(e.message || e));
    }

    let trendParsed = { headers: [], rows: [] };
    try {
      const tr = await fetchText(DATA.trendCsv);
      trendParsed = parseCsv(tr);
    } catch (e) {
      appendError("Tendencia CSV: " + String(e.message || e));
    }
    try {
      renderTrend(state, trendParsed);
    } catch (e) {
      appendError("Tendencia: " + String(e.message || e));
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDashboard);
  } else {
    initDashboard();
  }
})();
