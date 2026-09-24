/* Search over search.json, and the decision-log filter. No dependencies; both degrade to a
   plain page if this file fails to load. */
(function () {
  "use strict";

  var input = document.getElementById("q");
  var results = document.getElementById("results");
  var index = null;
  var selected = -1;

  function base() {
    return input.getAttribute("data-index").replace(/search\.json$/, "");
  }

  function load() {
    if (index !== null) return Promise.resolve(index);
    return fetch(input.getAttribute("data-index"))
      .then(function (r) { return r.json(); })
      .then(function (data) { index = data; return index; })
      .catch(function () { index = []; return index; });
  }

  function score(entry, needle) {
    var title = entry.t.toLowerCase();
    if (title === needle) return 100;
    if (title.indexOf(needle) === 0) return 80;
    if (title.indexOf(needle) > -1) return 60;
    for (var i = 0; i < entry.h.length; i++) {
      if (entry.h[i].t.toLowerCase().indexOf(needle) > -1) return 40 - i * 0.01;
    }
    if ((entry.x || "").toLowerCase().indexOf(needle) > -1) return 20;
    return 0;
  }

  function heading(entry, needle) {
    for (var i = 0; i < entry.h.length; i++) {
      if (entry.h[i].t.toLowerCase().indexOf(needle) > -1) return entry.h[i];
    }
    return null;
  }

  function render(needle) {
    if (!needle) { results.hidden = true; results.innerHTML = ""; return; }
    var hits = [];
    (index || []).forEach(function (entry) {
      var s = score(entry, needle);
      if (s > 0) hits.push({ e: entry, s: s, h: heading(entry, needle) });
    });
    hits.sort(function (a, b) { return b.s - a.s; });
    hits = hits.slice(0, 25);
    if (!hits.length) {
      results.innerHTML = '<div class="empty">Nothing matches “' + escape_(needle) + '”.</div>';
      results.hidden = false;
      return;
    }
    results.innerHTML = hits.map(function (hit, n) {
      var href = base() + hit.e.u + (hit.h ? "#" + hit.h.a : "");
      var sub = hit.h ? hit.h.t : (hit.e.x || "").slice(0, 110);
      return '<a href="' + href + '" data-n="' + n + '">' +
        '<span class="r-sec">' + escape_(hit.e.s || "") + "</span> " +
        "<b>" + escape_(hit.e.t) + "</b>" +
        '<div class="r-sub">' + escape_(sub) + "</div></a>";
    }).join("");
    results.hidden = false;
    selected = -1;
  }

  function escape_(text) {
    return String(text).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  if (input) {
    input.addEventListener("input", function () {
      var needle = input.value.trim().toLowerCase();
      if (needle.length < 2) { render(""); return; }
      load().then(function () { render(needle); });
    });
    input.addEventListener("keydown", function (event) {
      var links = results.querySelectorAll("a");
      if (event.key === "Escape") { render(""); input.blur(); return; }
      if (!links.length) return;
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        selected += event.key === "ArrowDown" ? 1 : -1;
        selected = Math.max(0, Math.min(links.length - 1, selected));
        links.forEach(function (a) { a.classList.remove("sel"); });
        links[selected].classList.add("sel");
        links[selected].scrollIntoView({ block: "nearest" });
      } else if (event.key === "Enter" && selected > -1) {
        event.preventDefault();
        window.location.href = links[selected].getAttribute("href");
      }
    });
    document.addEventListener("click", function (event) {
      if (!results.contains(event.target) && event.target !== input) render("");
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "/" && document.activeElement !== input) {
        event.preventDefault();
        input.focus();
      }
    });
  }

  var filter = document.getElementById("adr-filter");
  if (filter) {
    filter.addEventListener("input", function () {
      var needle = filter.value.trim().toLowerCase();
      var rows = document.querySelectorAll("#adr-table tbody tr");
      rows.forEach(function (row) {
        row.hidden = needle !== "" && row.textContent.toLowerCase().indexOf(needle) === -1;
      });
    });
  }

  if (window.renderMathInElement) {
    render_math();
  } else {
    window.addEventListener("load", function () {
      if (window.renderMathInElement) render_math();
    });
  }

  function render_math() {
    var targets = document.querySelectorAll(".prose, .toc, .page-head");
    targets.forEach(function (node) { math_in(node); });
  }

  function math_in(node) {
    window.renderMathInElement(node, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false }
      ],
      throwOnError: false,
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"]
    });
  }
})();
